interface SafetyBannerProps {
  hasRedFlags?: boolean
}

export default function SafetyBanner({ hasRedFlags = false }: SafetyBannerProps) {
  return (
    <div className="space-y-2">
      {/* Educational disclaimer - always visible */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 text-sm text-blue-800">
        <span className="font-medium">For demonstration purposes only:</span> This app does not provide
        medical advice, diagnosis, or treatment. Always consult a healthcare provider.
      </div>

      {/* Red flag warning - shown when red flags detected */}
      {hasRedFlags && (
        <div className="bg-red-50 border border-red-300 rounded-lg p-3 text-sm text-red-800">
          <span className="font-bold">Important:</span> If symptoms feel severe,
          rapidly worsening, or concerning, consider contacting a clinician promptly.
        </div>
      )}
    </div>
  )
}
