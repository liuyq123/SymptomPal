import { describe, it, expect } from 'vitest'
import type { DemoData, DemoLogMetadata } from '../types/demoPlayer'

const DEMO_FILES = ['frank_russo', 'elena_martinez', 'sarah_chen']
const REQUIRED_METADATA_FIELDS: (keyof DemoLogMetadata)[] = [
  'symptoms', 'actions_taken', 'red_flags', 'protocol', 'clinician_note',
  'safety_mode', 'reason_code', 'tool_calls',
]

describe('Demo data quality', () => {
  for (const patient of DEMO_FILES) {
    describe(patient, () => {
      it('all logs have complete trace metadata', async () => {
        const mod = await import(`../../public/demo/${patient}.json`)
        const data = mod.default as DemoData
        expect(data.logs.length).toBeGreaterThan(0)
        for (const log of data.logs) {
          for (const field of REQUIRED_METADATA_FIELDS) {
            expect(log.metadata, `Day ${log.day} missing ${field}`).toHaveProperty(field)
          }
        }
      })

      it('no contradictory protocol + reason_code', async () => {
        const mod = await import(`../../public/demo/${patient}.json`)
        const data = mod.default as DemoData
        for (const log of data.logs) {
          if (log.metadata.protocol) {
            expect(
              log.metadata.reason_code,
              `Day ${log.day}: has protocol but reason=no_protocol_match`
            ).not.toBe('no_protocol_match')
          }
        }
      })

      it('static-safety entries have empty tool_calls', async () => {
        const mod = await import(`../../public/demo/${patient}.json`)
        const data = mod.default as DemoData
        for (const log of data.logs) {
          if (log.metadata.safety_mode === 'static_safety') {
            expect(
              log.metadata.tool_calls,
              `Day ${log.day}: static_safety must have empty tool_calls`
            ).toEqual([])
          }
        }
      })

      it('has watchdog_results with clinician_observations', async () => {
        const mod = await import(`../../public/demo/${patient}.json`)
        const data = mod.default as DemoData
        expect(data.watchdog_results, 'missing watchdog_results').toBeDefined()
        expect(
          data.watchdog_results!.clinician_observations.length,
          'no clinician_observations'
        ).toBeGreaterThan(0)
      })
    })
  }
})
