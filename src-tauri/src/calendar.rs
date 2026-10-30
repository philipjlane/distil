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
    pub attendees: Vec<String>,
    /// Raw text from the event, as EventKit reports it. A conference link can
    /// be in any of these, so `video_link` scans them rather than the sidecar.
    #[serde(default)]
    pub location: Option<String>,
    #[serde(default)]
    pub notes: Option<String>,
    #[serde(default)]
    pub url: Option<String>,
}

/// Hosts whose links mean "this is a video call". A link matches when its host
/// is one of these or a subdomain of one (`us02web.zoom.us`).
const VIDEO_HOSTS: &[&str] = &[
    "zoom.us",
    "meet.google.com",
    "teams.microsoft.com",
    "teams.live.com",
    "webex.com",
    "meet.jit.si",
];

/// Outlook rewrites the links in invites into
/// `https://<region>.safelinks.protection.outlook.com/?url=<encoded>&data=...`.
const SAFE_LINKS_HOST: &str = "safelinks.protection.outlook.com";

impl CalendarEvent {
    /// The first video-conference link in the title, location, notes or URL,
    /// with an Outlook Safe Link unwrapped to the link it points at.
    pub fn video_link(&self) -> Option<String> {
        [
            Some(self.title.as_str()),
            self.location.as_deref(),
            self.notes.as_deref(),
            self.url.as_deref(),
        ]
        .into_iter()
        .flatten()
        .find_map(video_link_in)
    }
}

fn video_link_in(text: &str) -> Option<String> {
    text.split(|c: char| c.is_whitespace() || c == '<' || c == '>')
        .filter_map(clean_url)
        .find_map(|url| {
            let url = unwrap_safe_link(&url).unwrap_or(url);
            is_video_url(&url).then_some(url)
        })
}

/// The URL inside a whitespace-delimited word, shorn of the punctuation prose
/// wraps around links: `(https://zoom.us/j/1).` becomes `https://zoom.us/j/1`.
fn clean_url(word: &str) -> Option<String> {
    let lower = word.to_ascii_lowercase();
    let start = ["http://", "https://"]
        .iter()
        .filter_map(|scheme| lower.find(scheme))
        .min()?;
    let url =
        word[start..].trim_end_matches([')', ']', '}', '.', ',', ';', ':', '!', '?', '"', '\'']);
    Some(url.to_string())
}

/// Lower-cased host of a URL, without userinfo or port.
fn url_host(url: &str) -> Option<String> {
    let (_, rest) = url.split_once("://")?;
    let authority = rest.split(['/', '?', '#']).next()?;
    let host = authority.rsplit('@').next()?.split(':').next()?;
    (!host.is_empty()).then(|| host.to_ascii_lowercase())
}

fn host_is(host: &str, domain: &str) -> bool {
    host == domain
        || host
            .strip_suffix(domain)
            .is_some_and(|rest| rest.ends_with('.'))
}

fn is_video_url(url: &str) -> bool {
    url_host(url).is_some_and(|host| VIDEO_HOSTS.iter().any(|domain| host_is(&host, domain)))
}

/// The link an Outlook Safe Links wrapper points at, if `url` is one.
fn unwrap_safe_link(url: &str) -> Option<String> {
    if !host_is(&url_host(url)?, SAFE_LINKS_HOST) {
        return None;
    }
    let query = url.split_once('?')?.1.split('#').next()?;
    let target = percent_decode(
        query
            .split('&')
            .find_map(|pair| pair.strip_prefix("url="))?,
    );
    target
        .to_ascii_lowercase()
        .starts_with("http")
        .then_some(target)
}

/// Decodes `%XX` escapes (as UTF-8); anything malformed is kept as written.
fn percent_decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let hex = |offset: usize| {
            bytes
                .get(i + offset)
                .and_then(|b| (*b as char).to_digit(16))
        };
        match (bytes[i], hex(1), hex(2)) {
            (b'%', Some(high), Some(low)) => {
                out.push((high * 16 + low) as u8);
                i += 3;
            }
            (byte, ..) => {
                out.push(byte);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// A "real meeting" has other people in it: attendees on the invite (2+
/// including the user) or a video-conference link. Outlook can also leave the
/// link in the body and write only "Microsoft Teams Meeting" in the location,
/// so that counts too. Solo blocks, reminders and all-day events don't deserve
/// a recording prompt.
pub fn is_real_meeting(event: &CalendarEvent) -> bool {
    !event.is_all_day
        && (event.attendee_count >= 2
            || event.video_link().is_some()
            || event
                .location
                .as_deref()
                .is_some_and(|l| l.to_lowercase().contains("microsoft teams meeting")))
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
            attendees: Vec::new(),
            location: None,
            notes: link.map(|l| format!("Join: {l}")),
            url: None,
        }
    }

    fn with_notes(notes: &str) -> CalendarEvent {
        CalendarEvent {
            notes: Some(notes.into()),
            ..event("Sync", false, 0, None)
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
        let json = r#"{"title":"Standup","startsAt":"2026-08-04T09:30:00Z","isAllDay":false,"attendeeCount":0,"notes":"Join https://meet.google.com/abc","videoLink":"ignored"}"#;
        let event: CalendarEvent = serde_json::from_str(json).unwrap();
        assert_eq!(event.title, "Standup");
        assert_eq!(event.location, None);
        assert_eq!(
            event.video_link().as_deref(),
            Some("https://meet.google.com/abc")
        );
        assert!(is_real_meeting(&event));
    }

    #[test]
    fn video_links_are_found_in_every_field() {
        let mut e = event("Call https://zoom.us/j/1 now", false, 0, None);
        assert_eq!(e.video_link().as_deref(), Some("https://zoom.us/j/1"));
        e.title = "Sync".into();
        e.location = Some("https://meet.jit.si/room".into());
        assert_eq!(e.video_link().as_deref(), Some("https://meet.jit.si/room"));
        e.location = None;
        e.url = Some("https://acme.webex.com/meet/jo".into());
        assert_eq!(
            e.video_link().as_deref(),
            Some("https://acme.webex.com/meet/jo")
        );
        e.url = None;
        assert_eq!(e.video_link(), None);
    }

    #[test]
    fn teams_links_are_recognised() {
        let work = with_notes(
            "Join: <https://teams.microsoft.com/l/meetup-join/19%3ameeting_x%40thread.v2/0>",
        );
        assert_eq!(
            work.video_link().as_deref(),
            Some("https://teams.microsoft.com/l/meetup-join/19%3ameeting_x%40thread.v2/0")
        );
        let personal = with_notes("https://teams.live.com/meet/9412345678901");
        assert_eq!(
            personal.video_link().as_deref(),
            Some("https://teams.live.com/meet/9412345678901")
        );
        assert!(is_real_meeting(&personal));
    }

    #[test]
    fn outlook_safe_links_are_unwrapped() {
        let wrapped = with_notes(
            "Join <https://eur01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fteams.microsoft.com%2Fl%2Fmeetup-join%2F19%253ameeting_x%2540thread.v2%2F0&data=05%7C02%7C&reserved=0>",
        );
        assert_eq!(
            wrapped.video_link().as_deref(),
            Some("https://teams.microsoft.com/l/meetup-join/19%3ameeting_x%40thread.v2/0")
        );
        assert!(is_real_meeting(&wrapped));

        // url= need not be the first parameter, and a link to anything else stays ignored.
        let reordered = with_notes(
            "https://nam06.safelinks.protection.outlook.com/?data=05&url=https%3A%2F%2Fus02web.zoom.us%2Fj%2F123%3Fpwd%3Dabc&reserved=0",
        );
        assert_eq!(
            reordered.video_link().as_deref(),
            Some("https://us02web.zoom.us/j/123?pwd=abc")
        );
        let unrelated = with_notes(
            "https://eur01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fexample.com%2Fagenda&data=05",
        );
        assert_eq!(unrelated.video_link(), None);
        let broken = with_notes("https://eur01.safelinks.protection.outlook.com/?data=05");
        assert_eq!(broken.video_link(), None);
    }

    #[test]
    fn trailing_punctuation_is_not_part_of_the_link() {
        for (text, link) in [
            ("(https://zoom.us/j/1)", "https://zoom.us/j/1"),
            ("Join at https://zoom.us/j/1.", "https://zoom.us/j/1"),
            (
                "https://meet.google.com/abc-defg-hij, then",
                "https://meet.google.com/abc-defg-hij",
            ),
            ("\"https://zoom.us/j/1\"", "https://zoom.us/j/1"),
            ("'https://zoom.us/j/1'", "https://zoom.us/j/1"),
            (
                "<a href=\"https://zoom.us/j/1\">Join</a>",
                "https://zoom.us/j/1",
            ),
        ] {
            assert_eq!(
                with_notes(text).video_link().as_deref(),
                Some(link),
                "{text}"
            );
        }
    }

    #[test]
    fn only_real_video_hosts_count() {
        for text in [
            "https://notzoom.us/j/1",
            "https://example.com/?next=https://zoom.us/j/1",
            "https://zoom.us.evil.example/j/1",
            "zoom.us/j/1",
            "see teams.microsoft.com",
        ] {
            assert_eq!(with_notes(text).video_link(), None, "{text}");
        }
        assert_eq!(
            with_notes("HTTPS://Zoom.US/j/1").video_link().as_deref(),
            Some("HTTPS://Zoom.US/j/1")
        );
    }

    #[test]
    fn a_teams_location_alone_makes_a_meeting() {
        let mut e = event("Planning", false, 0, None);
        e.location = Some("Microsoft Teams Meeting".into());
        assert!(is_real_meeting(&e));
        e.location = Some("microsoft teams meeting".into());
        assert!(is_real_meeting(&e));
        e.location = Some("Boardroom".into());
        assert!(!is_real_meeting(&e));
        e.location = Some("Microsoft Teams Meeting".into());
        e.is_all_day = true;
        assert!(!is_real_meeting(&e));
    }

    #[test]
    fn percent_decoding_keeps_malformed_escapes() {
        assert_eq!(percent_decode("a%2Fb%3a%20c"), "a/b: c");
        assert_eq!(percent_decode("100%"), "100%");
        assert_eq!(percent_decode("%zz%4"), "%zz%4");
        assert_eq!(percent_decode("caf%C3%A9+x"), "caf\u{e9}+x");
    }
}
