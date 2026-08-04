use serde::Deserialize;

#[derive(Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct CalendarEvent {
    pub title: String,
    pub starts_at: String,
    #[serde(default)]
    pub is_all_day: bool,
    #[serde(default)]
    pub attendee_count: i64,
    #[serde(default)]
    pub video_link: Option<String>,
}

/// A "real meeting" has other people in it: attendees on the invite (2+
/// including the user) or a video-conference link. Solo blocks, reminders and
/// all-day events don't deserve a recording prompt.
pub fn is_real_meeting(event: &CalendarEvent) -> bool {
    !event.is_all_day && (event.attendee_count >= 2 || event.video_link.is_some())
}

/// Meeting titles become folder names; keep them filesystem-friendly.
pub fn sanitize_title(title: &str) -> String {
    let cleaned: String = title
        .chars()
        .map(|c| if c == '/' || c == ':' || c == '\\' || c == '\0' { ' ' } else { c })
        .collect();
    let joined = cleaned.split_whitespace().collect::<Vec<_>>().join(" ");
    let truncated: String = joined.chars().take(60).collect();
    let truncated = truncated.trim().to_string();
    if truncated.is_empty() { "Meeting".to_string() } else { truncated }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(title: &str, all_day: bool, attendees: i64, link: Option<&str>) -> CalendarEvent {
        CalendarEvent {
            title: title.into(),
            starts_at: "2026-08-04T14:00:00Z".into(),
            is_all_day: all_day,
            attendee_count: attendees,
            video_link: link.map(String::from),
        }
    }

    #[test]
    fn meetings_need_people_or_a_video_link() {
        assert!(is_real_meeting(&event("Weekly sync", false, 3, None)));
        assert!(is_real_meeting(&event("1:1", false, 0, Some("https://zoom.us/j/1"))));
        assert!(!is_real_meeting(&event("Focus block", false, 0, None)));
        assert!(!is_real_meeting(&event("Holiday", true, 5, None)));
        assert!(!is_real_meeting(&event("Solo invite", false, 1, None)));
    }

    #[test]
    fn titles_become_safe_folder_names() {
        assert_eq!(sanitize_title("Q3 review: budget / hiring"), "Q3 review budget hiring");
        assert_eq!(sanitize_title("   "), "Meeting");
        assert_eq!(sanitize_title(&"x".repeat(100)).chars().count(), 60);
    }

    #[test]
    fn calendar_events_parse_from_sidecar_json() {
        let json = r#"{"title":"Standup","startsAt":"2026-08-04T09:30:00Z","isAllDay":false,"attendeeCount":4,"videoLink":"https://meet.google.com/abc"}"#;
        let event: CalendarEvent = serde_json::from_str(json).unwrap();
        assert_eq!(event.title, "Standup");
        assert!(is_real_meeting(&event));
    }
}
