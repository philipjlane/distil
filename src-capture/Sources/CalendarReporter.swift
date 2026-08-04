import EventKit
import Foundation

/// Reads upcoming events from the macOS Calendar database (EventKit) — the
/// accounts the user already syncs there. No OAuth; nothing leaves the Mac.
enum CalendarReporter {
    static func runAndPrint(hours: Double) async {
        let store = EKEventStore()
        let granted = (try? await store.requestFullAccessToEvents()) ?? false
        guard granted else {
            Emit.event(["event": "calendar", "authorized": false, "events": [[String: Any]]()])
            exit(0)
        }

        let start = Date()
        let end = start.addingTimeInterval(hours * 3600)
        let predicate = store.predicateForEvents(withStart: start, end: end, calendars: nil)
        let formatter = ISO8601DateFormatter()

        let events = store.events(matching: predicate).map { event -> [String: Any] in
            var haystack = [event.title ?? "", event.location ?? "", event.notes ?? ""]
                .joined(separator: " ")
            if let url = event.url?.absoluteString {
                haystack += " " + url
            }
            return [
                "title": event.title ?? "Untitled",
                "startsAt": formatter.string(from: event.startDate),
                "endsAt": formatter.string(from: event.endDate),
                "isAllDay": event.isAllDay,
                "attendeeCount": event.attendees?.count ?? 0,
                "videoLink": firstVideoLink(in: haystack) as Any,
            ]
        }
        Emit.event(["event": "calendar", "authorized": true, "events": events])
        exit(0)
    }

    static func firstVideoLink(in text: String) -> String? {
        let hosts = ["zoom.us", "meet.google.com", "teams.microsoft.com", "webex.com", "meet.jit.si"]
        for word in text.split(whereSeparator: { $0.isWhitespace || $0 == "<" || $0 == ">" }) {
            let candidate = String(word)
            if candidate.hasPrefix("http"), hosts.contains(where: { candidate.contains($0) }) {
                return candidate
            }
        }
        return nil
    }
}
