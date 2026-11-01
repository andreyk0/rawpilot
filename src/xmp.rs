//! In-process reader and writer for the two XMP properties rawpilot owns: `xmp:Rating` and
//! `xmp:Label`.
//!
//! A sidecar belongs to RawTherapee and to whatever else touches it, so an edit changes only
//! the bytes of the property it sets. The file is scanned once with quick-xml's event reader,
//! which yields the byte span of every place a property is stored; the edit is then spliced
//! into the original bytes. Other properties, other namespaces, the `<?xpacket?>` wrapper and
//! its padding come through untouched, and a property keeps the form it already has:
//! exiv2 (RawTherapee) writes `xmp:Rating="3"` as an attribute of `rdf:Description`, exiftool
//! writes `<xmp:Rating>3</xmp:Rating>`, and other tools split properties over several
//! `rdf:Description` blocks. Properties are matched by namespace URI, never by prefix.
//!
//! A sidecar that is not well-formed XMP is an error, not an unrated image. A missing sidecar
//! is created as a minimal packet; an empty or whitespace-only file is overwritten the same
//! way, since it holds nothing to preserve.

use std::fs;
use std::io::ErrorKind;
use std::ops::Range;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use quick_xml::Reader;
use quick_xml::XmlVersion;
use quick_xml::escape::resolve_predefined_entity;
use quick_xml::events::{BytesStart, Event};
use thiserror::Error;

use crate::cache::write_atomic;
use crate::sidecar::{FoundMark, Mark, parse_xmp_label, xmp_label};

const XMP_NS: &str = "http://ns.adobe.com/xap/1.0/";
const RDF_NS: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";

/// The prefix a new declaration of the XMP namespace prefers.
const PREFERRED_PREFIX: &str = "xmp";

const UTF8_BOM: &[u8] = &[0xEF, 0xBB, 0xBF];

/// An empty packet: the wrapper and `rdf:RDF` that exiftool and exiv2 both produce, with no
/// `x:xmptk` so no tool is named as the author.
const SKELETON: &str = "<?xpacket begin=\"\u{feff}\" id=\"W5M0MpCehiHzreSzNTczkc9d\"?>\n\
<x:xmpmeta xmlns:x=\"adobe:ns:meta/\">\n\
<rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n\
</rdf:RDF>\n\
</x:xmpmeta>\n\
<?xpacket end=\"w\"?>\n";

/// Current rating and label of each sidecar, in the order given; `None` for a file that does
/// not exist.
pub fn read_marks(sidecars: &[PathBuf]) -> Result<Vec<Option<FoundMark>>> {
    sidecars
        .iter()
        .map(|path| {
            read_if_exists(path)?
                .map(|bytes| {
                    found_mark(&bytes).with_context(|| format!("reading {}", path.display()))
                })
                .transpose()
        })
        .collect()
}

/// Sets rating and label in each sidecar, creating it if needed and leaving every other
/// piece of content as it was. A `None` label removes the property. Each file is replaced
/// atomically; a failure leaves the files before it written and the rest untouched, and
/// running the batch again finishes it.
pub fn write_marks(writes: &[(PathBuf, Mark)]) -> Result<()> {
    for (path, mark) in writes {
        let before = read_if_exists(path)?;
        let after = with_mark(before.as_deref().unwrap_or_default(), *mark)
            .with_context(|| format!("updating {}", path.display()))?;
        write_atomic(path, &after).with_context(|| format!("writing {}", path.display()))?;
    }
    Ok(())
}

fn read_if_exists(path: &Path) -> Result<Option<Vec<u8>>> {
    match fs::read(path) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(e) if e.kind() == ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
    }
}

#[derive(Debug, Error)]
pub enum XmpError {
    #[error("malformed XML at byte {at}: {source}")]
    Xml { at: u64, source: quick_xml::Error },
    #[error("malformed XML: {0}")]
    Invalid(String),
    #[error("not an XMP packet: no rdf:RDF element")]
    NotXmp,
    #[error("xmp:{property} holds {value:?}, which is not {expected}")]
    BadValue {
        property: Property,
        value: String,
        expected: &'static str,
    },
    #[error("xmp:{property} appears with conflicting values {first:?} and {second:?}")]
    Conflict {
        property: Property,
        first: String,
        second: String,
    },
    #[error("xmp:{property} has child elements; expected a plain value")]
    NotPlain { property: Property },
    #[error("rdf:RDF is in the default namespace, so a rdf:Description cannot be added to it")]
    UnprefixedRdf,
}

/// The two properties rawpilot owns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Property {
    Rating,
    Label,
}

impl std::fmt::Display for Property {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.local_name())
    }
}

impl Property {
    const ALL: [Property; 2] = [Property::Rating, Property::Label];

    fn local_name(self) -> &'static str {
        match self {
            Property::Rating => "Rating",
            Property::Label => "Label",
        }
    }

    fn from_local_name(name: &str) -> Option<Property> {
        Property::ALL.into_iter().find(|p| p.local_name() == name)
    }
}

/// Where one stored occurrence of a property lives, as byte ranges into the source.
#[derive(Debug)]
enum Slot {
    /// `prefix:Name="value"` on an `rdf:Description`. `whole` includes the whitespace before
    /// it; `value` is between the quotes.
    Attribute {
        whole: Range<usize>,
        value: Range<usize>,
    },
    /// `<q>value</q>`; `content` is between the tags.
    Text {
        whole: Range<usize>,
        content: Range<usize>,
    },
    /// `<q/>`; `close` is the `/>`.
    Empty {
        whole: Range<usize>,
        close: Range<usize>,
        qname: String,
    },
}

/// One change to the source: `range` is replaced by `text`. An insertion has an empty range.
#[derive(Debug)]
struct Edit {
    range: Range<usize>,
    text: String,
}

impl Slot {
    fn set(&self, value: &str) -> Edit {
        match self {
            Slot::Attribute { value: r, .. } | Slot::Text { content: r, .. } => Edit {
                range: r.clone(),
                text: value.to_owned(),
            },
            Slot::Empty { close, qname, .. } => Edit {
                range: close.clone(),
                text: format!(">{value}</{qname}>"),
            },
        }
    }

    fn remove(&self, src: &[u8]) -> Edit {
        let range = match self {
            Slot::Attribute { whole, .. } => whole.clone(),
            Slot::Text { whole, .. } | Slot::Empty { whole, .. } => element_lines(src, whole),
        };
        Edit {
            range,
            text: String::new(),
        }
    }
}

/// `element` widened to the whole line when nothing else shares it, so removing it leaves no
/// blank line; otherwise widened over its leading indentation only.
fn element_lines(src: &[u8], element: &Range<usize>) -> Range<usize> {
    let indent_start = src[..element.start]
        .iter()
        .rposition(|b| !matches!(b, b' ' | b'\t'))
        .map_or(0, |i| i + 1);
    let alone_before = indent_start == 0 || src[indent_start - 1] == b'\n';
    let line_break = match &src[element.end..] {
        [b'\r', b'\n', ..] => 2,
        [b'\n', ..] => 1,
        _ => 0,
    };
    if alone_before && line_break > 0 {
        indent_start..element.end + line_break
    } else {
        indent_start..element.end
    }
}

#[derive(Debug)]
struct Occurrence {
    property: Property,
    /// Index into `Scan::descriptions`.
    description: usize,
    slot: Slot,
    value: String,
}

impl Occurrence {
    fn form(&self) -> Form {
        match self.slot {
            Slot::Attribute { .. } => Form::Attribute,
            Slot::Text { .. } | Slot::Empty { .. } => Form::Element,
        }
    }
}

/// How a property is written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Form {
    Attribute,
    Element,
}

/// An element new properties can be added to: its start tag, and the whitespace that
/// separates its children.
#[derive(Debug)]
struct Host {
    /// Where a new attribute (or namespace declaration) goes: the end of the start tag, before
    /// any whitespace that precedes `>` or `/>`.
    attribute_at: usize,
    shape: Shape,
    /// The whitespace between the start tag and the first child, when the first child is on
    /// its own line.
    leading_whitespace: Option<String>,
}

#[derive(Debug)]
enum Shape {
    /// `close` is the start of the `/>`.
    SelfClosing { close: usize },
    /// Children go right after the start tag, which ends at `content_start`.
    Open { content_start: usize },
}

impl Host {
    /// Children are inserted at the start of the content, indented like the existing first
    /// child; `fallback` when there is none to copy.
    fn child_separator(&self, fallback: &str) -> String {
        self.leading_whitespace
            .clone()
            .unwrap_or_else(|| fallback.to_owned())
    }
}

#[derive(Debug)]
struct Description {
    host: Host,
    /// A prefix bound to the XMP namespace where this element sits, if any.
    xmp_prefix: Option<String>,
    /// A prefix not bound to anything where this element sits, for declaring the namespace.
    free_prefix: String,
}

#[derive(Debug)]
struct Scan {
    occurrences: Vec<Occurrence>,
    descriptions: Vec<Description>,
    /// The `rdf:RDF` element and the prefix its own name carries.
    rdf: Host,
    rdf_prefix: Option<String>,
}

impl Scan {
    fn of(&self, property: Property) -> impl Iterator<Item = &Occurrence> {
        self.occurrences
            .iter()
            .filter(move |o| o.property == property)
    }

    fn value(&self, property: Property) -> Result<Option<&str>, XmpError> {
        let mut values = self
            .of(property)
            .map(|o| o.value.as_str())
            .filter(|v| !v.trim().is_empty());
        let Some(first) = values.next() else {
            return Ok(None);
        };
        match values.find(|v| *v != first) {
            Some(second) => Err(XmpError::Conflict {
                property,
                first: first.to_owned(),
                second: second.to_owned(),
            }),
            None => Ok(Some(first)),
        }
    }
}

/// Rating and label as stored in `src`.
pub fn found_mark(src: &[u8]) -> Result<FoundMark, XmpError> {
    if is_blank(src) {
        return Ok(FoundMark::default());
    }
    let scan = Scan::parse(src)?;
    let rating = scan
        .value(Property::Rating)?
        .map(|v| {
            v.trim().parse::<i64>().map_err(|_| XmpError::BadValue {
                property: Property::Rating,
                value: v.to_owned(),
                expected: "an integer",
            })
        })
        .transpose()?;
    let label = scan.value(Property::Label)?.map(parse_xmp_label);
    Ok(FoundMark { rating, label })
}

/// `src` with rating and label set to `mark`. An empty `src` is a new sidecar.
pub fn with_mark(src: &[u8], mark: Mark) -> Result<Vec<u8>, XmpError> {
    if is_blank(src) {
        return with_mark(SKELETON.as_bytes(), mark);
    }
    let scan = Scan::parse(src)?;
    let targets = [
        (Property::Rating, Some(mark.rating.get().to_string())),
        (Property::Label, mark.label.map(|l| xmp_label(l).to_owned())),
    ];
    let mut edits: Vec<Edit> = Vec::new();
    let mut missing: Vec<(Property, String)> = Vec::new();
    for (property, target) in targets {
        let mut present = scan.of(property).peekable();
        match target {
            Some(value) if present.peek().is_some() => {
                edits.extend(present.map(|o| o.slot.set(&value)));
            }
            Some(value) => missing.push((property, value)),
            None => edits.extend(present.map(|o| o.slot.remove(src))),
        }
    }
    if !missing.is_empty() {
        edits.extend(scan.insertion(&missing)?);
    }
    Ok(splice(src, edits))
}

fn is_blank(src: &[u8]) -> bool {
    src.iter().all(u8::is_ascii_whitespace)
}

/// Applies non-overlapping edits. An insertion at byte `b` lands before a removal or
/// replacement that starts at `b`, so the two never collide.
fn splice(src: &[u8], mut edits: Vec<Edit>) -> Vec<u8> {
    edits.sort_by_key(|e| (e.range.start, e.range.end));
    let mut out = Vec::with_capacity(src.len() + 64);
    let mut at = 0;
    for e in &edits {
        assert!(at <= e.range.start, "overlapping edits at byte {at}: {e:?}");
        out.extend_from_slice(&src[at..e.range.start]);
        out.extend_from_slice(e.text.as_bytes());
        at = e.range.end;
    }
    out.extend_from_slice(&src[at..]);
    out
}

impl Scan {
    /// The edits that add `missing` where the file already keeps the properties, in the form
    /// they are kept in; else to its first `rdf:Description`; else in a new one.
    fn insertion(&self, missing: &[(Property, String)]) -> Result<Vec<Edit>, XmpError> {
        let sibling = self.occurrences.first();
        let description = sibling
            .map(|o| o.description)
            .or_else(|| (!self.descriptions.is_empty()).then_some(0));
        let Some(index) = description else {
            return self.new_description(missing);
        };
        let d = &self.descriptions[index];
        let (prefix, declaration) = match &d.xmp_prefix {
            Some(p) => (p.as_str(), String::new()),
            None => (
                d.free_prefix.as_str(),
                format!(" xmlns:{}=\"{XMP_NS}\"", d.free_prefix),
            ),
        };
        let sibling_form = sibling
            .filter(|o| o.description == index)
            .map(Occurrence::form);
        Ok(match (&d.host.shape, sibling_form) {
            (Shape::Open { content_start }, None | Some(Form::Element)) => {
                let separator = d.host.child_separator("\n  ");
                let elements: String = missing
                    .iter()
                    .map(|(p, v)| element(&separator, prefix, *p, v))
                    .collect();
                let mut edits = vec![Edit {
                    range: empty_at(*content_start),
                    text: elements,
                }];
                if !declaration.is_empty() {
                    edits.push(Edit {
                        range: empty_at(d.host.attribute_at),
                        text: declaration,
                    });
                }
                edits
            }
            (Shape::SelfClosing { .. }, _) | (Shape::Open { .. }, Some(Form::Attribute)) => {
                let attributes: String = missing
                    .iter()
                    .map(|(p, v)| format!(" {prefix}:{}=\"{v}\"", p.local_name()))
                    .collect();
                vec![Edit {
                    range: empty_at(d.host.attribute_at),
                    text: declaration + &attributes,
                }]
            }
        })
    }

    /// A new `rdf:Description` as the last child of `rdf:RDF`.
    fn new_description(&self, missing: &[(Property, String)]) -> Result<Vec<Edit>, XmpError> {
        let rdf = self.rdf_prefix.as_deref().ok_or(XmpError::UnprefixedRdf)?;
        let separator = self.rdf.child_separator("\n");
        let elements: String = missing
            .iter()
            .map(|(p, v)| element(&format!("{separator}  "), PREFERRED_PREFIX, *p, v))
            .collect();
        let description = format!(
            "{separator}<{rdf}:Description {rdf}:about=\"\" xmlns:{PREFERRED_PREFIX}=\"{XMP_NS}\">\
             {elements}{separator}</{rdf}:Description>"
        );
        Ok(match self.rdf.shape {
            Shape::Open { content_start } => vec![Edit {
                range: empty_at(content_start),
                text: description,
            }],
            Shape::SelfClosing { close } => vec![Edit {
                range: close..close + "/>".len(),
                text: format!(">{description}{separator}</{rdf}:RDF>"),
            }],
        })
    }
}

fn empty_at(at: usize) -> Range<usize> {
    at..at
}

fn element(separator: &str, prefix: &str, property: Property, value: &str) -> String {
    let name = format!("{prefix}:{}", property.local_name());
    format!("{separator}<{name}>{value}</{name}>")
}

/// A namespace declaration in scope.
struct Binding {
    prefix: Option<String>,
    uri: String,
}

/// What the enclosing element is, as far as a child's classification needs.
#[derive(Clone, Copy)]
enum Parent {
    None,
    Rdf,
    Description(usize),
    Other,
}

/// What an open element is, as far as the scan cares.
enum Role {
    Rdf,
    Description(usize),
    /// A property element whose text is still being read.
    Property(OpenProperty),
    Other,
}

struct OpenProperty {
    property: Property,
    description: usize,
    start: usize,
    content_start: usize,
    text: String,
}

struct Frame {
    role: Role,
    /// How many bindings were in scope outside this element.
    bindings_outside: usize,
}

/// A start or empty tag and where it sits in the source.
struct Tag {
    start: usize,
    end: usize,
    qname: String,
    is_empty_element: bool,
}

impl Scan {
    fn parse(src: &[u8]) -> Result<Scan, XmpError> {
        Parser::new(src).run()
    }
}

struct Parser<'a> {
    src: &'a [u8],
    reader: Reader<&'a [u8]>,
    /// Bytes of byte-order mark the reader skips, which its positions do not count.
    bom_len: usize,
    bindings: Vec<Binding>,
    stack: Vec<Frame>,
    occurrences: Vec<Occurrence>,
    descriptions: Vec<Description>,
    rdf: Option<(Host, Option<String>)>,
    /// The host whose first child has not been seen yet, with its start-tag end.
    awaiting_first_child: Option<AwaitingChild>,
}

/// Tracks the whitespace after a host's start tag, to copy its indentation.
struct AwaitingChild {
    target: HostRef,
    content_start: usize,
}

#[derive(Clone, Copy)]
enum HostRef {
    Rdf,
    Description(usize),
}

impl<'a> Parser<'a> {
    fn new(src: &'a [u8]) -> Self {
        Parser {
            src,
            reader: Reader::from_reader(src.strip_prefix(UTF8_BOM).unwrap_or(src)),
            bom_len: if src.starts_with(UTF8_BOM) {
                UTF8_BOM.len()
            } else {
                0
            },
            bindings: Vec::new(),
            stack: Vec::new(),
            occurrences: Vec::new(),
            descriptions: Vec::new(),
            rdf: None,
            awaiting_first_child: None,
        }
    }

    fn run(mut self) -> Result<Scan, XmpError> {
        loop {
            let start = self.position();
            let event = self.reader.read_event().map_err(|source| XmpError::Xml {
                at: self.reader.error_position(),
                source,
            })?;
            let end = self.position();
            match event {
                Event::Start(e) => self.open(&e, start, end, false)?,
                Event::Empty(e) => self.open(&e, start, end, true)?,
                Event::End(_) => self.close(start, end),
                Event::Text(t) => self.text(&t.xml10_content()),
                Event::CData(c) => self.text(&c.xml10_content()),
                Event::GeneralRef(r) => {
                    let text = match r.resolve_char_ref() {
                        Ok(Some(c)) => c.to_string(),
                        Ok(None) => resolve_predefined_entity(&r)
                            .ok_or_else(|| XmpError::Invalid(format!("unknown entity &{};", &*r)))?
                            .to_owned(),
                        Err(e) => return Err(XmpError::Invalid(e.to_string())),
                    };
                    self.text(&text);
                }
                Event::Eof => break,
                Event::Comment(_) | Event::Decl(_) | Event::PI(_) | Event::DocType(_) => {}
            }
        }
        if !self.stack.is_empty() {
            return Err(XmpError::Invalid("unclosed element at end of file".into()));
        }
        let (rdf, rdf_prefix) = self.rdf.ok_or(XmpError::NotXmp)?;
        Ok(Scan {
            occurrences: self.occurrences,
            descriptions: self.descriptions,
            rdf,
            rdf_prefix,
        })
    }

    fn position(&self) -> usize {
        self.bom_len
            + usize::try_from(self.reader.buffer_position()).expect("file offset fits in usize")
    }

    fn lookup(&self, prefix: Option<&str>) -> Option<&str> {
        self.bindings
            .iter()
            .rev()
            .find(|b| b.prefix.as_deref() == prefix)
            .map(|b| b.uri.as_str())
    }

    /// A prefix bound to the XMP namespace and not shadowed since.
    fn xmp_prefix(&self) -> Option<String> {
        let mut seen: Vec<Option<&str>> = Vec::new();
        for b in self.bindings.iter().rev() {
            if seen.contains(&b.prefix.as_deref()) {
                continue;
            }
            seen.push(b.prefix.as_deref());
            if b.uri == XMP_NS && b.prefix.is_some() {
                return b.prefix.clone();
            }
        }
        None
    }

    fn free_prefix(&self) -> String {
        std::iter::once(PREFERRED_PREFIX.to_owned())
            .chain((1..).map(|n| format!("{PREFERRED_PREFIX}{n}")))
            .find(|p| self.lookup(Some(p)).is_none())
            .expect("an unbounded sequence has a free element")
    }

    /// The namespace of an element name, which an unprefixed name takes from the default.
    fn element_ns(&self, qname: &str) -> Result<(Option<String>, String), XmpError> {
        match qname.split_once(':') {
            Some((prefix, local)) => {
                let uri = self.lookup(Some(prefix)).ok_or_else(|| {
                    XmpError::Invalid(format!("undeclared namespace prefix {prefix:?}"))
                })?;
                Ok((Some(uri.to_owned()), local.to_owned()))
            }
            None => Ok((self.lookup(None).map(str::to_owned), qname.to_owned())),
        }
    }

    fn open(
        &mut self,
        tag: &BytesStart<'_>,
        start: usize,
        end: usize,
        is_empty_element: bool,
    ) -> Result<(), XmpError> {
        self.note_child(start);
        let parent = match self.stack.last().map(|f| &f.role) {
            None => Parent::None,
            Some(Role::Rdf) => Parent::Rdf,
            Some(Role::Description(d)) => Parent::Description(*d),
            Some(Role::Property(p)) => {
                return Err(XmpError::NotPlain {
                    property: p.property,
                });
            }
            Some(Role::Other) => Parent::Other,
        };
        let bindings_outside = self.bindings.len();
        let attributes = self.declare(tag)?;
        let tag = Tag {
            start,
            end,
            qname: tag.name().as_ref().to_owned(),
            is_empty_element,
        };
        let (uri, local) = self.element_ns(&tag.qname)?;
        let role = match (uri.as_deref(), local.as_str(), parent) {
            (Some(RDF_NS), "RDF", _) if self.rdf.is_none() => {
                self.begin_rdf(&tag);
                Role::Rdf
            }
            (Some(RDF_NS), "Description", Parent::Rdf) => {
                Role::Description(self.begin_description(&tag, &attributes))
            }
            (Some(XMP_NS), name, Parent::Description(d)) => Property::from_local_name(name)
                .map_or(Role::Other, |property| {
                    self.begin_property(&tag, property, d)
                }),
            _ => Role::Other,
        };
        if tag.is_empty_element {
            self.bindings.truncate(bindings_outside);
        } else {
            self.stack.push(Frame {
                role,
                bindings_outside,
            });
        }
        Ok(())
    }

    /// Pushes the namespace declarations of `tag` and returns its other attributes.
    fn declare(&mut self, tag: &BytesStart<'_>) -> Result<Vec<RawAttribute>, XmpError> {
        let mut others = Vec::new();
        for attribute in tag.attributes() {
            let attribute = attribute.map_err(|e| XmpError::Invalid(e.to_string()))?;
            let key: &str = attribute.key.as_ref();
            let value = attribute
                .normalized_value(XmlVersion::Implicit1_0)
                .map_err(|e| XmpError::Invalid(e.to_string()))?
                .into_owned();
            if key == "xmlns" {
                self.bindings.push(Binding {
                    prefix: None,
                    uri: value,
                });
            } else if let Some(prefix) = key.strip_prefix("xmlns:") {
                self.bindings.push(Binding {
                    prefix: Some(prefix.to_owned()),
                    uri: value,
                });
            } else {
                others.push(RawAttribute {
                    key: key.to_owned(),
                    key_start: offset_in(self.src, key),
                    value,
                    raw_value_len: attribute.value.len(),
                });
            }
        }
        Ok(others)
    }

    fn begin_rdf(&mut self, tag: &Tag) {
        let host = self.host(tag);
        let prefix = tag.qname.split_once(':').map(|(p, _)| p.to_owned());
        self.rdf = Some((host, prefix));
        self.await_child(HostRef::Rdf, tag);
    }

    fn begin_description(&mut self, tag: &Tag, attributes: &[RawAttribute]) -> usize {
        let index = self.descriptions.len();
        self.descriptions.push(Description {
            host: self.host(tag),
            xmp_prefix: self.xmp_prefix(),
            free_prefix: self.free_prefix(),
        });
        self.await_child(HostRef::Description(index), tag);
        for a in attributes {
            let Some((prefix, local)) = a.key.split_once(':') else {
                continue;
            };
            let Some(property) = Property::from_local_name(local) else {
                continue;
            };
            if self.lookup(Some(prefix)) != Some(XMP_NS) {
                continue;
            }
            // The value starts after the quote that follows `=`; XML permits whitespace around it.
            let key_end = a.key_start + a.key.len();
            let quote = self.src[key_end..]
                .iter()
                .position(|b| matches!(b, b'"' | b'\''))
                .expect("an attribute value is quoted");
            let value_start = key_end + quote + 1;
            let value = value_start..value_start + a.raw_value_len;
            let whole_start = self.src[..a.key_start]
                .iter()
                .rposition(|b| !b.is_ascii_whitespace())
                .map_or(0, |i| i + 1);
            self.occurrences.push(Occurrence {
                property,
                description: index,
                slot: Slot::Attribute {
                    whole: whole_start..value.end + 1,
                    value,
                },
                value: a.value.clone(),
            });
        }
        index
    }

    fn begin_property(&mut self, tag: &Tag, property: Property, description: usize) -> Role {
        if tag.is_empty_element {
            self.occurrences.push(Occurrence {
                property,
                description,
                slot: Slot::Empty {
                    whole: tag.start..tag.end,
                    close: tag.end - "/>".len()..tag.end,
                    qname: tag.qname.clone(),
                },
                value: String::new(),
            });
            return Role::Other;
        }
        Role::Property(OpenProperty {
            property,
            description,
            start: tag.start,
            content_start: tag.end,
            text: String::new(),
        })
    }

    /// The start tag of the element `tag`.
    fn host(&self, tag: &Tag) -> Host {
        let closing = if tag.is_empty_element { "/>" } else { ">" };
        let before_closing = tag.end - closing.len();
        let attribute_at = self.src[..before_closing]
            .iter()
            .rposition(|b| !b.is_ascii_whitespace())
            .map_or(0, |i| i + 1);
        Host {
            attribute_at,
            shape: if tag.is_empty_element {
                Shape::SelfClosing {
                    close: before_closing,
                }
            } else {
                Shape::Open {
                    content_start: tag.end,
                }
            },
            leading_whitespace: None,
        }
    }

    fn await_child(&mut self, target: HostRef, tag: &Tag) {
        self.awaiting_first_child = (!tag.is_empty_element).then_some(AwaitingChild {
            target,
            content_start: tag.end,
        });
    }

    /// An element start closes the wait for a first child; the whitespace before it, if the
    /// text between is only whitespace, is its indentation.
    fn note_child(&mut self, child_start: usize) {
        let Some(waiting) = self.awaiting_first_child.take() else {
            return;
        };
        let between = &self.src[waiting.content_start..child_start];
        let Ok(between) = std::str::from_utf8(between) else {
            return;
        };
        if between.chars().all(char::is_whitespace) && between.contains('\n') {
            let host = match waiting.target {
                HostRef::Rdf => self.rdf.as_mut().map(|(h, _)| h),
                HostRef::Description(i) => Some(&mut self.descriptions[i].host),
            };
            if let Some(host) = host {
                host.leading_whitespace = Some(between.to_owned());
            }
        }
    }

    fn text(&mut self, text: &str) {
        if let Some(Frame {
            role: Role::Property(open),
            ..
        }) = self.stack.last_mut()
        {
            open.text.push_str(text);
        }
    }

    fn close(&mut self, start: usize, end: usize) {
        let frame = self.stack.pop().expect("quick-xml checks end-tag nesting");
        self.bindings.truncate(frame.bindings_outside);
        self.awaiting_first_child = None;
        if let Role::Property(open) = frame.role {
            self.occurrences.push(Occurrence {
                property: open.property,
                description: open.description,
                slot: Slot::Text {
                    whole: open.start..end,
                    content: open.content_start..start,
                },
                value: open.text,
            });
        }
    }
}

struct RawAttribute {
    key: String,
    key_start: usize,
    /// Unescaped.
    value: String,
    /// Length of the value as written, between the quotes.
    raw_value_len: usize,
}

/// Where `part`, a slice borrowed from `src`, starts in `src`.
fn offset_in(src: &[u8], part: &str) -> usize {
    let offset = (part.as_ptr() as usize)
        .checked_sub(src.as_ptr() as usize)
        .expect("slice borrowed from the source");
    assert!(
        offset + part.len() <= src.len(),
        "slice borrowed from the source"
    );
    offset
}

#[cfg(test)]
mod tests {
    //! Fixtures are real sidecars from exiftool and exiv2 (`tests/fixtures/xmp/`) plus
    //! hand-written ones for the layouts those tools do not produce. Expected output is the
    //! fixture text with the one edited span replaced by hand, so a test fails if anything
    //! else in the file moves. `exiftool_agrees_with_our_writes` (ignored; needs exiftool and
    //! exiv2) compares a write against both tools' reading of the file.

    use std::process::Command;

    use super::*;
    use crate::rate::{Label, Stars};
    use crate::sidecar::FoundLabel;

    const EXIFTOOL: &str = include_str!("../tests/fixtures/xmp/exiftool_elements.xmp");
    const EXIV2: &str = include_str!("../tests/fixtures/xmp/exiv2_attributes.xmp");
    const SELF_CLOSING: &str = include_str!("../tests/fixtures/xmp/self_closing_negative.xmp");
    const CUSTOM_PREFIX: &str = include_str!("../tests/fixtures/xmp/custom_prefix_split.xmp");
    const OTHER_NS: &str = include_str!("../tests/fixtures/xmp/other_namespaces.xmp");

    fn mark(rating: u8, label: Option<Label>) -> Mark {
        Mark {
            rating: Stars::new(rating),
            label,
        }
    }

    fn found(rating: i64, label: FoundLabel) -> FoundMark {
        FoundMark {
            rating: Some(rating),
            label: Some(label),
        }
    }

    fn written(src: &str, m: Mark) -> String {
        String::from_utf8(with_mark(src.as_bytes(), m).unwrap()).unwrap()
    }

    /// `text` with `old` replaced by `new`, which must occur exactly once.
    fn swap(text: &str, old: &str, new: &str) -> String {
        assert_eq!(text.matches(old).count(), 1, "{old:?} in fixture");
        text.replacen(old, new, 1)
    }

    #[test]
    fn reads_element_form() {
        assert_eq!(
            found_mark(EXIFTOOL.as_bytes()).unwrap(),
            found(3, FoundLabel::Known(Label::Green))
        );
    }

    #[test]
    fn reads_attribute_form_beside_other_namespaces() {
        assert_eq!(
            found_mark(EXIV2.as_bytes()).unwrap(),
            found(3, FoundLabel::Known(Label::Green))
        );
    }

    #[test]
    fn reads_negative_rating_from_a_self_closing_description() {
        assert_eq!(
            found_mark(SELF_CLOSING.as_bytes()).unwrap(),
            found(-1, FoundLabel::Known(Label::Red))
        );
    }

    #[test]
    fn reads_by_namespace_not_prefix_across_descriptions() {
        // Rating is `adobe:` in one block, Label is `a:` with an entity in a third, and a
        // block that binds `xmp:` to another namespace holds decoys.
        assert_eq!(
            found_mark(CUSTOM_PREFIX.as_bytes()).unwrap(),
            found(4, FoundLabel::Other("Blue & Gold".into()))
        );
    }

    #[test]
    fn reads_nothing_when_the_marks_are_absent() {
        assert_eq!(
            found_mark(OTHER_NS.as_bytes()).unwrap(),
            FoundMark::default()
        );
    }

    #[test]
    fn changes_only_the_rating_and_label_text_in_element_form() {
        let want = swap(EXIFTOOL, "<xmp:Rating>3<", "<xmp:Rating>5<");
        let want = swap(&want, ">Green<", ">Yellow<");
        assert_eq!(written(EXIFTOOL, mark(5, Some(Label::Yellow))), want);
    }

    #[test]
    fn changes_only_the_values_in_attribute_form() {
        let want = swap(EXIV2, "xmp:Rating=\"3\"", "xmp:Rating=\"1\"");
        let want = swap(&want, "xmp:Label=\"Green\"", "xmp:Label=\"Purple\"");
        assert_eq!(written(EXIV2, mark(1, Some(Label::Purple))), want);
    }

    #[test]
    fn writing_the_stored_values_changes_nothing() {
        let m = mark(3, Some(Label::Green));
        assert_eq!(written(EXIFTOOL, m), EXIFTOOL);
        assert_eq!(written(EXIV2, m), EXIV2);
    }

    #[test]
    fn replaces_a_negative_rating_in_a_self_closing_description() {
        let want = swap(SELF_CLOSING, "xmp:Rating=\"-1\"", "xmp:Rating=\"2\"");
        assert_eq!(written(SELF_CLOSING, mark(2, Some(Label::Red))), want);
    }

    #[test]
    fn edits_each_description_that_holds_a_property_and_no_decoy() {
        let want = swap(CUSTOM_PREFIX, "<adobe:Rating>4<", "<adobe:Rating>2<");
        let want = swap(&want, "a:Label=\"Blue &amp; Gold\"", "a:Label=\"Green\"");
        assert_eq!(written(CUSTOM_PREFIX, mark(2, Some(Label::Green))), want);
    }

    #[test]
    fn removing_the_label_deletes_its_element_line() {
        let want = swap(EXIFTOOL, "  <xmp:Label>Green</xmp:Label>\n", "");
        assert_eq!(written(EXIFTOOL, mark(3, None)), want);
    }

    #[test]
    fn removing_the_label_deletes_its_attribute() {
        let want = swap(EXIV2, "\n   xmp:Label=\"Green\"", "");
        assert_eq!(written(EXIV2, mark(3, None)), want);
    }

    #[test]
    fn removing_an_absent_label_changes_only_the_rating() {
        let src = swap(EXIFTOOL, "  <xmp:Label>Green</xmp:Label>\n", "");
        assert_eq!(written(&src, mark(4, None)), swap(&src, ">3<", ">4<"));
    }

    #[test]
    fn adds_elements_to_a_description_with_children_declaring_the_namespace() {
        let want = swap(
            OTHER_NS,
            "crs:Exposure2012=\"+0.35\">\n",
            "crs:Exposure2012=\"+0.35\" xmlns:xmp=\"http://ns.adobe.com/xap/1.0/\">\n\
             \x20  <xmp:Rating>3</xmp:Rating>\n\
             \x20  <xmp:Label>Green</xmp:Label>\n",
        );
        assert_eq!(written(OTHER_NS, mark(3, Some(Label::Green))), want);
    }

    #[test]
    fn adds_a_missing_label_as_an_attribute_beside_an_attribute_rating() {
        let src = swap(EXIV2, "\n   xmp:Label=\"Green\"", "");
        let want = swap(
            &src,
            "xmp:Rating=\"3\"",
            "xmp:Rating=\"3\" xmp:Label=\"Blue\"",
        );
        assert_eq!(written(&src, mark(3, Some(Label::Blue))), want);
    }

    #[test]
    fn adds_a_missing_rating_as_an_element_beside_an_element_label() {
        let src = swap(EXIFTOOL, "  <xmp:Rating>3</xmp:Rating>\n", "");
        let want = swap(
            &src,
            "  <xmp:Label>Green</xmp:Label>\n",
            "  <xmp:Rating>2</xmp:Rating>\n  <xmp:Label>Green</xmp:Label>\n",
        );
        assert_eq!(written(&src, mark(2, Some(Label::Green))), want);
    }

    #[test]
    fn adds_attributes_to_a_self_closing_description_using_its_prefix() {
        let src = swap(SELF_CLOSING, "\n   xmp:Label=\"Red\"", "");
        let want = swap(
            &src,
            "xmp:Rating=\"-1\"",
            "xmp:Rating=\"1\" xmp:Label=\"Red\"",
        );
        assert_eq!(written(&src, mark(1, Some(Label::Red))), want);
    }

    #[test]
    fn declares_a_prefix_that_does_not_clash_with_one_in_scope() {
        let src = "<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"><rdf:RDF \
                   xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\" \
                   xmlns:xmp=\"urn:other\"><rdf:Description rdf:about=\"\"/></rdf:RDF></x:xmpmeta>";
        let out = written(src, mark(4, None));
        assert_eq!(
            out,
            src.replace(
                "rdf:about=\"\"/>",
                "rdf:about=\"\" xmlns:xmp1=\"http://ns.adobe.com/xap/1.0/\" xmp1:Rating=\"4\"/>"
            )
        );
        assert_eq!(found_mark(out.as_bytes()).unwrap().rating, Some(4));
    }

    #[test]
    fn adds_a_description_to_an_rdf_without_one() {
        let src = "<rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\"/>";
        assert_eq!(
            written(src, mark(5, Some(Label::Red))),
            "<rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n\
             <rdf:Description rdf:about=\"\" xmlns:xmp=\"http://ns.adobe.com/xap/1.0/\">\n\
             \x20 <xmp:Rating>5</xmp:Rating>\n\
             \x20 <xmp:Label>Red</xmp:Label>\n\
             </rdf:Description>\n\
             </rdf:RDF>"
        );
    }

    const RDF_OPEN: &str = "<rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">";
    const DESCRIPTION_OPEN: &str =
        "<rdf:Description rdf:about=\"\" xmlns:xmp=\"http://ns.adobe.com/xap/1.0/\">";

    #[test]
    fn an_insertion_beside_a_removal_at_the_same_byte_does_not_collide() {
        // The label is the first child on the start tag's line and there is no rating: the
        // rating is inserted at the byte where the label's removal starts.
        for label in ["<xmp:Label>Red</xmp:Label>", "<xmp:Label/>"] {
            let src = format!("{RDF_OPEN}{DESCRIPTION_OPEN}{label}</rdf:Description></rdf:RDF>");
            let want = format!(
                "{RDF_OPEN}{DESCRIPTION_OPEN}\n  <xmp:Rating>3</xmp:Rating></rdf:Description></rdf:RDF>"
            );
            assert_eq!(written(&src, mark(3, None)), want, "{label}");
        }
    }

    #[test]
    fn a_blank_file_becomes_a_minimal_packet() {
        let expected = "<?xpacket begin=\"\u{feff}\" id=\"W5M0MpCehiHzreSzNTczkc9d\"?>\n\
            <x:xmpmeta xmlns:x=\"adobe:ns:meta/\">\n\
            <rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n\
            <rdf:Description rdf:about=\"\" xmlns:xmp=\"http://ns.adobe.com/xap/1.0/\">\n\
            \x20 <xmp:Rating>3</xmp:Rating>\n\
            \x20 <xmp:Label>Green</xmp:Label>\n\
            </rdf:Description>\n\
            </rdf:RDF>\n\
            </x:xmpmeta>\n\
            <?xpacket end=\"w\"?>\n";
        let m = mark(3, Some(Label::Green));
        assert_eq!(written("", m), expected);
        assert_eq!(written("\n  \n", m), expected);
        assert_eq!(found_mark(expected.as_bytes()).unwrap(), FoundMark::from(m));
    }

    #[test]
    fn closes_a_self_closing_rdf_that_has_whitespace_before_the_slash() {
        let src = "<rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\"  />";
        let out = written(src, mark(1, None));
        assert!(out.starts_with(
            "<rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\"  >\n"
        ));
        assert!(out.ends_with("</rdf:Description>\n</rdf:RDF>"));
        assert_eq!(found_mark(out.as_bytes()).unwrap().rating, Some(1));
    }

    #[test]
    fn offsets_stay_correct_after_a_byte_order_mark() {
        let src = format!("\u{feff}{EXIFTOOL}");
        let want = format!("\u{feff}{}", swap(EXIFTOOL, ">3<", ">4<"));
        assert_eq!(written(&src, mark(4, Some(Label::Green))), want);
    }

    #[test]
    fn rejects_bytes_that_are_not_utf8() {
        let utf16: Vec<u8> = EXIFTOOL.encode_utf16().flat_map(u16::to_le_bytes).collect();
        assert!(found_mark(&utf16).is_err());
        assert!(with_mark(&utf16, mark(1, None)).is_err());
    }

    #[test]
    fn a_blank_file_reads_as_unmarked() {
        assert_eq!(found_mark(b"").unwrap(), FoundMark::default());
    }

    #[test]
    fn every_target_round_trips_through_every_fixture_and_is_stable() {
        let fixtures = [EXIFTOOL, EXIV2, SELF_CLOSING, CUSTOM_PREFIX, OTHER_NS, ""];
        for src in fixtures {
            for rating in 0..=5 {
                for label in [None, Some(Label::Red), Some(Label::Purple)] {
                    let m = mark(rating, label);
                    let once = written(src, m);
                    assert_eq!(found_mark(once.as_bytes()).unwrap(), FoundMark::from(m));
                    assert_eq!(written(&once, m), once, "second write moves nothing");
                }
            }
        }
    }

    /// Every fixture as is, on one line with single spaces between lines, on one line with no
    /// whitespace between elements (so a property can be the first thing after a start tag),
    /// and with CRLF line ends; plus the exiftool fixture without its rating in tight form.
    fn layouts() -> Vec<(String, String)> {
        let mut all = Vec::new();
        let mut tight_exiftool = String::new();
        for (name, src) in [
            ("exiftool", EXIFTOOL),
            ("exiv2", EXIV2),
            ("self_closing", SELF_CLOSING),
            ("custom_prefix", CUSTOM_PREFIX),
            ("other_ns", OTHER_NS),
        ] {
            let spaced = src.lines().map(str::trim).collect::<Vec<_>>().join(" ");
            let tight = spaced.replace("> <", "><");
            if name == "exiftool" {
                tight_exiftool.clone_from(&tight);
            }
            all.push((format!("{name} as is"), src.to_owned()));
            all.push((format!("{name} spaced"), spaced));
            all.push((format!("{name} tight"), tight));
            all.push((format!("{name} crlf"), src.replace('\n', "\r\n")));
        }
        let no_rating = swap(&tight_exiftool, "<xmp:Rating>3</xmp:Rating>", "");
        all.push((
            "exiftool tight with an empty label".to_owned(),
            swap(&no_rating, "<xmp:Label>Green</xmp:Label>", "<xmp:Label/>"),
        ));
        all.push(("exiftool tight without a rating".to_owned(), no_rating));
        all
    }

    /// Goal: `with_mark` is total over damaged and re-laid-out documents. Method: every
    /// layout of every fixture, cut at every byte, written with a rating alone and with
    /// rating and label; the result is `Ok` or `Err`, never a panic, and an `Ok` document
    /// reads back as the mark written.
    #[test]
    fn writing_never_panics_and_reads_back() {
        let marks = [mark(3, None), mark(5, Some(Label::Yellow))];
        for (name, src) in layouts() {
            for cut in 0..=src.len() {
                let Some(prefix) = src.get(..cut) else {
                    continue;
                };
                for m in marks {
                    if let Ok(out) = with_mark(prefix.as_bytes(), m) {
                        assert_eq!(
                            found_mark(&out).unwrap(),
                            FoundMark::from(m),
                            "{name} cut at {cut}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn rejects_content_that_is_not_xmp() {
        for (name, bad) in [
            ("plain text", "not xmp at all <<<".to_owned()),
            (
                "no rdf",
                "<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"></x:xmpmeta>".to_owned(),
            ),
            ("truncated", EXIFTOOL[..EXIFTOOL.len() / 2].to_owned()),
            (
                "mismatched tags",
                swap(EXIFTOOL, "</xmp:Label>", "</xmp:Rating>"),
            ),
            ("unclosed", swap(EXIFTOOL, "</x:xmpmeta>", "")),
            (
                "undeclared prefix",
                swap(EXIFTOOL, "<xmp:Label>", "<nope:Label>"),
            ),
        ] {
            assert!(found_mark(bad.as_bytes()).is_err(), "read: {name}");
            assert!(
                with_mark(bad.as_bytes(), mark(1, None)).is_err(),
                "write: {name}"
            );
        }
    }

    #[test]
    fn rejects_values_it_cannot_interpret_as_marks() {
        let word = swap(EXIFTOOL, ">3<", ">many<");
        let bad = found_mark(word.as_bytes()).unwrap_err();
        assert!(matches!(
            bad,
            XmpError::BadValue {
                property: Property::Rating,
                ..
            }
        ));
        assert_eq!(
            bad.to_string(),
            "xmp:Rating holds \"many\", which is not an integer"
        );
        let nested = swap(EXIFTOOL, ">3<", "><rdf:Alt/><");
        assert!(matches!(
            found_mark(nested.as_bytes()),
            Err(XmpError::NotPlain { .. })
        ));
        let twice = swap(
            EXIFTOOL,
            "  <xmp:Label>",
            "  <xmp:Rating>4</xmp:Rating>\n  <xmp:Label>",
        );
        assert!(matches!(
            found_mark(twice.as_bytes()),
            Err(XmpError::Conflict { .. })
        ));
    }

    #[test]
    fn a_duplicated_property_is_set_everywhere() {
        let twice = swap(
            EXIFTOOL,
            "  <xmp:Label>",
            "  <xmp:Rating>3</xmp:Rating>\n  <xmp:Label>",
        );
        let out = written(&twice, mark(5, Some(Label::Green)));
        assert_eq!(out.matches("<xmp:Rating>5<").count(), 2);
    }

    #[test]
    fn reads_an_empty_element_as_absent_and_fills_it() {
        let src = swap(EXIFTOOL, "<xmp:Rating>3</xmp:Rating>", "<xmp:Rating/>");
        assert_eq!(found_mark(src.as_bytes()).unwrap().rating, None);
        let want = swap(&src, "<xmp:Rating/>", "<xmp:Rating>2</xmp:Rating>");
        assert_eq!(written(&src, mark(2, Some(Label::Green))), want);
    }

    #[test]
    fn files_are_read_written_and_created_on_disk() {
        let dir = tempfile::tempdir().unwrap();
        let existing = dir.path().join("a.xmp");
        let created = dir.path().join("b.xmp");
        let missing = dir.path().join("c.xmp");
        fs::write(&existing, EXIV2).unwrap();
        let m = mark(4, Some(Label::Blue));
        write_marks(&[(existing.clone(), m), (created.clone(), m)]).unwrap();

        let got = read_marks(&[existing.clone(), created.clone(), missing.clone()]).unwrap();
        assert_eq!(
            got,
            [Some(FoundMark::from(m)), Some(FoundMark::from(m)), None]
        );
        assert_eq!(
            fs::read_to_string(&existing).unwrap(),
            swap(
                &swap(EXIV2, "xmp:Rating=\"3\"", "xmp:Rating=\"4\""),
                "\"Green\"",
                "\"Blue\""
            )
        );
        assert!(!dir.path().join("a.xmp.tmp").exists());
    }

    #[test]
    fn a_corrupt_sidecar_is_an_error_and_is_left_alone() {
        let dir = tempfile::tempdir().unwrap();
        let bad = dir.path().join("bad.xmp");
        fs::write(&bad, "garbage").unwrap();
        assert!(read_marks(std::slice::from_ref(&bad)).is_err());
        assert!(write_marks(&[(bad.clone(), mark(1, None))]).is_err());
        assert_eq!(fs::read_to_string(&bad).unwrap(), "garbage");
    }

    type Tags = serde_json::Map<String, serde_json::Value>;

    /// `exiftool -j -a -G1` of `path` without the file-system, file-type and exiftool-version entries.
    fn exiftool_tags(path: &Path) -> Tags {
        let out = Command::new("exiftool")
            .args(["-j", "-a", "-G1"])
            .arg(path)
            .output()
            .expect("exiftool is installed");
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let mut all: Vec<Tags> = serde_json::from_slice(&out.stdout).unwrap();
        let mut tags = all.remove(0);
        tags.retain(|k, _| {
            !(k.starts_with("System:")
                || k.starts_with("File:")
                || k.starts_with("ExifTool:")
                || k == "SourceFile")
        });
        tags
    }

    fn exiv2_value(path: &Path, key: &str) -> Option<String> {
        let out = Command::new("exiv2")
            .arg("-px")
            .arg(path)
            .output()
            .expect("exiv2 is installed");
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .find(|l| l.starts_with(key))
            .map(|l| l.split_whitespace().skip(3).collect::<Vec<_>>().join(" "))
    }

    #[test]
    #[ignore = "needs exiftool and exiv2 on PATH"]
    fn exiftool_agrees_with_our_writes() {
        let dir = tempfile::tempdir().unwrap();
        let cases: [(&str, &str); 6] = [
            ("exiftool", EXIFTOOL),
            ("exiv2", EXIV2),
            ("self_closing", SELF_CLOSING),
            ("custom_prefix", CUSTOM_PREFIX),
            ("other_ns", OTHER_NS),
            ("created", ""),
        ];
        for (name, src) in cases {
            for m in [mark(5, Some(Label::Yellow)), mark(0, None)] {
                let path = dir.path().join(format!("{name}-{}.xmp", m.rating.get()));
                let mut before = Tags::new();
                if !src.is_empty() {
                    fs::write(&path, src).unwrap();
                    before = exiftool_tags(&path);
                }
                write_marks(&[(path.clone(), m)]).unwrap();
                let mut after = exiftool_tags(&path);

                let rating = after.remove("XMP-xmp:Rating");
                let label = after.remove("XMP-xmp:Label");
                assert_eq!(rating, Some(m.rating.get().into()), "{name} rating");
                assert_eq!(
                    label.as_ref().and_then(|l| l.as_str()),
                    m.label.map(xmp_label),
                    "{name} label"
                );
                before.remove("XMP-xmp:Rating");
                before.remove("XMP-xmp:Label");
                assert_eq!(before, after, "{name}: every other tag unchanged");

                // exiv2's XMP toolkit rejects the XMP namespace under any prefix but `xmp`
                // ("Schema namespace URI and prefix mismatch"), so it cannot read this
                // fixture at all, before or after our write.
                if name == "custom_prefix" {
                    continue;
                }

                assert_eq!(
                    exiv2_value(&path, "Xmp.xmp.Rating"),
                    Some(m.rating.get().to_string()),
                    "{name} exiv2 rating"
                );
                assert_eq!(
                    exiv2_value(&path, "Xmp.xmp.Label"),
                    m.label.map(|l| xmp_label(l).to_owned()),
                    "{name} exiv2 label"
                );
            }
        }
    }
}
