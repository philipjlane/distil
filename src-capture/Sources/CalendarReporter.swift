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
            var fields: [String: Any] = [
                "title": event.title ?? "Untitled",
                "startsAt": formatter.string(from: event.startDate),
                "endsAt": formatter.string(from: event.endDate),
                "isAllDay": event.isAllDay,
                "attendeeCount": event.attendees?.count ?? 0,
                "attendees": event.attendees?.compactMap { $0.name } ?? [String](),
            ]
            // Raw text only: the host looks for a video-conference link in these
            // (calendar.rs), where it is testable.
            if let location = event.location { fields["location"] = location }
            if let notes = event.notes { fields["notes"] = notes }
            if let url = event.url?.absoluteString { fields["url"] = url }
            return fields
        }
        Emit.event(["event": "calendar", "authorized": true, "events": events])
        exit(0)
    }
}
