"""Integration tests for API routes."""

import pytest
import os
import asyncio
from datetime import datetime, timedelta, timezone
import httpx

# Keep route tests deterministic and fast.
os.environ["USE_STUB_MEDGEMMA"] = "true"
os.environ["USE_LOCAL_MEDGEMMA"] = "false"
os.environ["USE_STUB_RESPONSE_GEN"] = "true"
os.environ["USE_PROTOCOL_FOLLOWUP"] = "true"
os.environ["PROTOCOL_SHADOW_MODE"] = "false"
os.environ["PRELOAD_HEAR"] = "false"
os.environ["USE_STUB_HEAR"] = "true"

from ..main import app
from ..models import ScheduledCheckin, CheckinType
from ..services.storage import create_scheduled_checkin


class SyncASGIClient:
    """Synchronous test client backed by httpx ASGI transport.

    We avoid fastapi.TestClient here because it hangs in this environment.
    """

    def __init__(self, asgi_app):
        self._app = asgi_app
        self.headers = {}

    def request(self, method: str, url: str, **kwargs):
        async def _run():
            transport = httpx.ASGITransport(app=self._app)
            headers = dict(self.headers)
            request_headers = kwargs.pop("headers", None)
            if request_headers:
                headers.update(request_headers)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
                headers=headers,
            ) as client:
                return await client.request(method, url, **kwargs)

        return asyncio.run(_run())

    def get(self, url: str, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs):
        return self.request("POST", url, **kwargs)

    def patch(self, url: str, **kwargs):
        return self.request("PATCH", url, **kwargs)

    def delete(self, url: str, **kwargs):
        return self.request("DELETE", url, **kwargs)


@pytest.fixture
def client(test_user_id):
    """Create a test client with auth headers."""
    os.environ["API_KEY"] = "test_key"
    client = SyncASGIClient(app)
    client.headers.update({"X-API-Key": "test_key", "X-User-Id": test_user_id})
    return client


@pytest.fixture
def test_user_id():
    """Unique user ID for each test run."""
    return f"test_user_{datetime.now(timezone.utc).replace(tzinfo=None).timestamp()}"


class TestIngestRoutes:
    """Tests for /api/ingest endpoints."""

    def test_ingest_voice_with_text(self, client, test_user_id):
        """Test ingesting text description."""
        response = client.post(
            "/api/ingest/voice",
            json={
                "user_id": test_user_id,
                "recorded_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
                "description_text": "I have a headache that started this morning",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "log" in data
        assert data["log"]["user_id"] == test_user_id
        assert "headache" in data["log"]["transcript"].lower()

    def test_ingest_voice_no_input_returns_400(self, client, test_user_id):
        """Test that missing input returns 400."""
        response = client.post(
            "/api/ingest/voice",
            json={
                "user_id": test_user_id,
                "recorded_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
            },
        )
        assert response.status_code == 400
        assert "At least one of" in response.json()["detail"]

    def test_ingest_voice_input_too_large(self, client, test_user_id):
        """Test that oversized input is rejected."""
        # Create a description larger than 10KB
        large_text = "a" * (11 * 1024)
        response = client.post(
            "/api/ingest/voice",
            json={
                "user_id": test_user_id,
                "recorded_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
                "description_text": large_text,
            },
        )
        assert response.status_code == 422  # Validation error


class TestLogRoutes:
    """Tests for /api/logs endpoints."""

    def test_list_logs_empty(self, client, test_user_id):
        """Test listing logs for user with no logs."""
        response = client.get(f"/api/logs?user_id={test_user_id}")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_logs_after_ingest(self, client, test_user_id):
        """Test listing logs after creating one."""
        # Create a log first
        client.post(
            "/api/ingest/voice",
            json={
                "user_id": test_user_id,
                "recorded_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
                "description_text": "Test symptom for log listing",
            },
        )

        # List logs
        response = client.get(f"/api/logs?user_id={test_user_id}")
        assert response.status_code == 200
        logs = response.json()
        assert len(logs) >= 1
        assert logs[0]["user_id"] == test_user_id

    def test_get_log_not_found(self, client):
        """Test getting a non-existent log."""
        response = client.get("/api/logs/nonexistent_log_id")
        assert response.status_code == 404


class TestSummarizeRoutes:
    """Tests for /api/summarize endpoints."""

    def test_doctor_packet_empty_user(self, client, test_user_id):
        """Test doctor packet generation for user with no logs."""
        response = client.post(
            "/api/summarize/doctor-packet",
            json={"user_id": test_user_id, "days": 7},
        )
        assert response.status_code == 200
        data = response.json()
        assert "hpi" in data
        assert "pertinent_positives" in data

    def test_timeline_empty_user(self, client, test_user_id):
        """Test timeline generation for user with no logs."""
        response = client.post(
            "/api/summarize/timeline",
            json={"user_id": test_user_id, "days": 7},
        )
        assert response.status_code == 200
        data = response.json()
        assert "story_points" in data

    def test_summarize_days_validation(self, client, test_user_id):
        """Test that invalid days parameter is rejected."""
        response = client.post(
            "/api/summarize/doctor-packet",
            json={"user_id": test_user_id, "days": -1},
        )
        assert response.status_code == 422

        response = client.post(
            "/api/summarize/doctor-packet",
            json={"user_id": test_user_id, "days": 400},
        )
        assert response.status_code == 422

    def test_summarize_days_zero_accepted(self, client, test_user_id):
        """Test that days=0 (all history) is accepted."""
        response = client.post(
            "/api/summarize/doctor-packet",
            json={"user_id": test_user_id, "days": 0},
        )
        assert response.status_code == 200


class TestMedicationRoutes:
    """Tests for /api/medications endpoints."""

    def test_list_medications_empty(self, client, test_user_id):
        """Test listing medications for user with none."""
        response = client.get(f"/api/medications?user_id={test_user_id}")
        assert response.status_code == 200
        assert response.json() == []

    def test_create_medication(self, client, test_user_id):
        """Test creating a medication."""
        response = client.post(
            "/api/medications",
            json={
                "user_id": test_user_id,
                "name": "Ibuprofen",
                "dose": "400mg",
                "frequency": "as needed",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Ibuprofen"
        assert data["dose"] == "400mg"
        assert data["is_active"] is True

    def test_update_medication_reminders(self, client, test_user_id):
        """Test enabling medication reminders and saving reminder times."""
        created = client.post(
            "/api/medications",
            json={
                "user_id": test_user_id,
                "name": "Reminder Test Med",
            },
        )
        assert created.status_code == 200
        med_id = created.json()["id"]

        updated = client.patch(
            f"/api/medications/{med_id}",
            json={
                "reminder_enabled": True,
                "reminder_times": ["08:00", "20:30"],
            },
        )
        assert updated.status_code == 200
        payload = updated.json()
        assert payload["reminder_enabled"] is True
        assert payload["reminder_times"] == ["08:00", "20:30"]

    def test_create_and_list_medication(self, client, test_user_id):
        """Test creating and then listing medications."""
        # Create
        client.post(
            "/api/medications",
            json={
                "user_id": test_user_id,
                "name": "Test Medication",
            },
        )

        # List
        response = client.get(f"/api/medications?user_id={test_user_id}")
        assert response.status_code == 200
        meds = response.json()
        assert len(meds) >= 1
        assert any(m["name"] == "Test Medication" for m in meds)

    def test_log_medication(self, client, test_user_id):
        """Test logging a medication dose."""
        response = client.post(
            "/api/medications/log",
            json={
                "user_id": test_user_id,
                "medication_name": "Ad-hoc medication",
                "dose_taken": "1 tablet",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["medication_name"] == "Ad-hoc medication"

    def test_medication_history(self, client, test_user_id):
        """Test getting medication history."""
        # Log a dose
        client.post(
            "/api/medications/log",
            json={
                "user_id": test_user_id,
                "medication_name": "Test med for history",
            },
        )

        # Get history
        response = client.get(f"/api/medications/log/history?user_id={test_user_id}")
        assert response.status_code == 200
        history = response.json()
        assert len(history) >= 1


class TestAmbientRoutes:
    """Tests for /api/ambient endpoints."""

    def test_start_session(self, client, test_user_id):
        """Test starting an ambient session."""
        response = client.post(
            "/api/ambient/sessions/start",
            json={
                "user_id": test_user_id,
                "session_type": "cough_monitor",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "session" in data
        assert data["session"]["status"] == "active"
        assert data["session"]["session_type"] == "cough_monitor"
        assert "upload_interval_seconds" in data

    def test_start_session_when_active_exists(self, client, test_user_id):
        """Test that starting a second session fails."""
        # Start first session
        client.post(
            "/api/ambient/sessions/start",
            json={
                "user_id": test_user_id,
                "session_type": "cough_monitor",
            },
        )

        # Try to start second session
        response = client.post(
            "/api/ambient/sessions/start",
            json={
                "user_id": test_user_id,
                "session_type": "sleep",
            },
        )
        assert response.status_code == 409
        assert "active session" in response.json()["detail"].lower()

    def test_list_sessions_empty(self, client, test_user_id):
        """Test listing sessions for user with none."""
        response = client.get(f"/api/ambient/sessions?user_id={test_user_id}")
        assert response.status_code == 200
        assert response.json() == []

    def test_get_active_session_none(self, client, test_user_id):
        """Test getting active session when none exists."""
        response = client.get(f"/api/ambient/sessions/active?user_id={test_user_id}")
        assert response.status_code == 200
        assert response.json() is None

    def test_get_active_session_exists(self, client, test_user_id):
        """Test getting active session when one exists."""
        # Start a session
        client.post(
            "/api/ambient/sessions/start",
            json={
                "user_id": test_user_id,
                "session_type": "sleep",
            },
        )

        # Get active session
        response = client.get(f"/api/ambient/sessions/active?user_id={test_user_id}")
        assert response.status_code == 200
        data = response.json()
        assert data is not None
        assert data["status"] == "active"

    def test_cancel_session(self, client, test_user_id):
        """Test cancelling an active session."""
        # Start a session
        start_response = client.post(
            "/api/ambient/sessions/start",
            json={
                "user_id": test_user_id,
                "session_type": "cough_monitor",
            },
        )
        session_id = start_response.json()["session"]["id"]

        # Cancel it
        response = client.post(
            f"/api/ambient/sessions/{session_id}/cancel?user_id={test_user_id}"
        )
        assert response.status_code == 200
        assert response.json()["status"] == "cancelled"

    def test_cancel_session_wrong_user(self, client, test_user_id):
        """Test that wrong user cannot cancel session."""
        # Start a session
        start_response = client.post(
            "/api/ambient/sessions/start",
            json={
                "user_id": test_user_id,
                "session_type": "cough_monitor",
            },
        )
        session_id = start_response.json()["session"]["id"]

        # Try to cancel with wrong user
        response = client.post(
            f"/api/ambient/sessions/{session_id}/cancel?user_id=wrong_user",
            headers={"X-API-Key": "test_key", "X-User-Id": "wrong_user"},
        )
        assert response.status_code == 403

    def test_upload_chunk(self, client, test_user_id):
        """Test uploading an audio chunk."""
        # Start a session
        start_response = client.post(
            "/api/ambient/sessions/start",
            json={
                "user_id": test_user_id,
                "session_type": "cough_monitor",
            },
        )
        session_id = start_response.json()["session"]["id"]

        # Upload a chunk (minimal base64 audio)
        response = client.post(
            "/api/ambient/sessions/upload",
            json={
                "session_id": session_id,
                "user_id": test_user_id,
                "chunk_index": 0,
                "audio_b64": "SGVsbG8gV29ybGQ=",  # "Hello World" in base64
                "duration_seconds": 30.0,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "chunk_id" in data
        assert "events_detected" in data

    def test_upload_chunk_wrong_session(self, client, test_user_id):
        """Test that uploading to non-existent session fails."""
        response = client.post(
            "/api/ambient/sessions/upload",
            json={
                "session_id": "nonexistent_session",
                "user_id": test_user_id,
                "chunk_index": 0,
                "audio_b64": "SGVsbG8=",
                "duration_seconds": 30.0,
            },
        )
        assert response.status_code == 404

    def test_end_session(self, client, test_user_id):
        """Test ending a session and getting results."""
        # Start a session
        start_response = client.post(
            "/api/ambient/sessions/start",
            json={
                "user_id": test_user_id,
                "session_type": "cough_monitor",
            },
        )
        session_id = start_response.json()["session"]["id"]

        # End it
        response = client.post(
            "/api/ambient/sessions/end",
            json={
                "session_id": session_id,
                "user_id": test_user_id,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "session" in data
        assert data["session"]["status"] == "completed"
        assert "result" in data
        assert "summary" in data["result"]


class TestCheckinRoutes:
    """Tests for /api/checkins lifecycle behavior."""

    def test_triggered_unanswered_checkin_still_pending(self, client, test_user_id):
        checkin_id = f"checkin_{datetime.now(timezone.utc).replace(tzinfo=None).timestamp()}".replace(".", "")
        create_scheduled_checkin(
            ScheduledCheckin(
                id=checkin_id,
                user_id=test_user_id,
                checkin_type=CheckinType.SYMPTOM_PROGRESSION,
                scheduled_for=datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=1),
                message="How are your symptoms now?",
                context={},
                created_at=datetime.now(timezone.utc).replace(tzinfo=None),
            )
        )

        pending = client.get(f"/api/checkins/pending?user_id={test_user_id}")
        assert pending.status_code == 200
        assert any(c["id"] == checkin_id for c in pending.json())

        mark_triggered = client.post(f"/api/checkins/{checkin_id}/trigger")
        assert mark_triggered.status_code == 200

        pending_after_trigger = client.get(f"/api/checkins/pending?user_id={test_user_id}")
        assert pending_after_trigger.status_code == 200
        assert any(c["id"] == checkin_id for c in pending_after_trigger.json())

    def test_responded_or_dismissed_checkin_not_pending(self, client, test_user_id):
        respond_id = f"checkin_resp_{datetime.now(timezone.utc).replace(tzinfo=None).timestamp()}".replace(".", "")
        dismiss_id = f"checkin_dismiss_{datetime.now(timezone.utc).replace(tzinfo=None).timestamp()}".replace(".", "")

        for checkin_id in [respond_id, dismiss_id]:
            create_scheduled_checkin(
                ScheduledCheckin(
                    id=checkin_id,
                    user_id=test_user_id,
                    checkin_type=CheckinType.SYMPTOM_PROGRESSION,
                    scheduled_for=datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=1),
                    message=f"Follow-up check-in {checkin_id}",
                    context={},
                    created_at=datetime.now(timezone.utc).replace(tzinfo=None),
                )
            )

        responded = client.post(
            f"/api/checkins/{respond_id}/respond",
            json={"response": "Feeling better now."},
        )
        assert responded.status_code == 200

        dismissed = client.post(f"/api/checkins/{dismiss_id}/dismiss")
        assert dismissed.status_code == 200

        pending = client.get(f"/api/checkins/pending?user_id={test_user_id}")
        assert pending.status_code == 200
        pending_ids = {item["id"] for item in pending.json()}
        assert respond_id not in pending_ids
        assert dismiss_id not in pending_ids


class TestProfileIntakeRoutes:
    """Tests for first-run profile intake flow."""

    def test_first_log_creates_profile_intake_checkin(self, client, test_user_id):
        ingested = client.post(
            "/api/ingest/voice",
            json={
                "user_id": test_user_id,
                "recorded_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
                "description_text": "new user headache log",
            },
        )
        assert ingested.status_code == 200

        pending = client.get(f"/api/checkins/pending?user_id={test_user_id}")
        assert pending.status_code == 200
        assert any(item["checkin_type"] == "profile_intake" for item in pending.json())

    def test_profile_intake_response_updates_profile_and_queues_next(self, client, test_user_id):
        ingested = client.post(
            "/api/ingest/voice",
            json={
                "user_id": test_user_id,
                "recorded_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
                "description_text": "first symptom log for profile intake",
            },
        )
        assert ingested.status_code == 200

        # Answer demographic questions first (name, age_sex)
        for expected_answer in ["Test User", "30, female"]:
            pending = client.get(f"/api/checkins/pending?user_id={test_user_id}")
            assert pending.status_code == 200
            intake = next(
                (item for item in pending.json() if item["checkin_type"] == "profile_intake"),
                None,
            )
            assert intake is not None
            responded = client.post(
                f"/api/checkins/{intake['id']}/respond",
                json={"response": expected_answer},
            )
            assert responded.status_code == 200

        # Now the conditions question should be pending
        pending_before = client.get(f"/api/checkins/pending?user_id={test_user_id}")
        assert pending_before.status_code == 200
        intake = next(
            (item for item in pending_before.json() if item["checkin_type"] == "profile_intake"),
            None,
        )
        assert intake is not None

        responded = client.post(
            f"/api/checkins/{intake['id']}/respond",
            json={"response": "Asthma and migraines"},
        )
        assert responded.status_code == 200

        profile = client.get(f"/api/profile?user_id={test_user_id}")
        assert profile.status_code == 200
        profile_payload = profile.json()
        assert profile_payload["intake_questions_asked"] >= 3
        # Demographics should be set directly
        assert profile_payload["name"] == "Test User"
        assert profile_payload["age"] == 30
        assert profile_payload["gender"] == "female"
        # Stub MedGemma can't parse intake — raw response deferred, not regex-parsed
        assert profile_payload["intake_pending_raw"].get("conditions") == "Asthma and migraines"

        pending_after = client.get(f"/api/checkins/pending?user_id={test_user_id}")
        assert pending_after.status_code == 200
        assert any(item["checkin_type"] == "profile_intake" for item in pending_after.json())


class TestHealthEndpoints:
    """Tests for health and root endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data

    def test_health_endpoint(self, client):
        """Test health endpoint returns healthy."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestInputValidation:
    """Tests for input validation limits."""

    def test_user_id_too_long(self, client):
        """Test that overly long user_id is rejected."""
        long_user_id = "a" * 300
        response = client.post(
            "/api/ingest/voice",
            json={
                "user_id": long_user_id,
                "recorded_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
                "description_text": "test",
            },
            headers={"X-API-Key": "test_key", "X-User-Id": long_user_id},
        )
        assert response.status_code == 422

    def test_chunk_index_negative(self, client, test_user_id):
        """Test that negative chunk_index is rejected."""
        # Start a session first
        start_response = client.post(
            "/api/ambient/sessions/start",
            json={
                "user_id": test_user_id,
                "session_type": "cough_monitor",
            },
        )
        session_id = start_response.json()["session"]["id"]

        response = client.post(
            "/api/ambient/sessions/upload",
            json={
                "session_id": session_id,
                "user_id": test_user_id,
                "chunk_index": -1,
                "audio_b64": "SGVsbG8=",
                "duration_seconds": 30.0,
            },
        )
        assert response.status_code == 422

    def test_duration_too_long(self, client, test_user_id):
        """Test that overly long duration is rejected."""
        # Start a session first
        start_response = client.post(
            "/api/ambient/sessions/start",
            json={
                "user_id": test_user_id,
                "session_type": "cough_monitor",
            },
        )
        session_id = start_response.json()["session"]["id"]

        response = client.post(
            "/api/ambient/sessions/upload",
            json={
                "session_id": session_id,
                "user_id": test_user_id,
                "chunk_index": 0,
                "audio_b64": "SGVsbG8=",
                "duration_seconds": 400.0,  # Max is 300
            },
        )
        assert response.status_code == 422
