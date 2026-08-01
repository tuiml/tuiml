# TuiML Website Design System ("oc")

The design language used across tuiml.ai, extracted from the landing page so
every page can share it. Terminal-flavored: one monospace font, flat surfaces,
hairline rules, ASCII bracket markers instead of icon fonts.

## Files

| File | Purpose |
|------|---------|
| `static/css/oc.css` | The whole design system: palette, layout, type, components |
| `static/js/oc.js` | Shared behaviors: tab strips, code-block copy buttons, scroll-reveal terminals, `OC.esc`/`OC.mdLite` helpers |
| `templates/components/_head.html` | Meta, fonts, favicon (include in every page `<head>`) |
| `templates/components/_site_nav.html` | The one navbar (set `active_nav` before including) |
| `templates/components/_footer.html` | The one footer |

## Using it on a page

```html
<body class="landing-page antialiased overflow-x-hidden">
{% set active_nav = 'projects' %}
{% include 'components/_site_nav.html' %}

<link rel="stylesheet" href="/static/css/oc.css">
<script src="/static/js/oc.js"></script>   <!-- no defer: pages use OC.* inline -->

<div class="oc-wrap">
    <section class="oc-section" style="padding-top: 64px;">  <!-- first section -->
        <h1 class="oc-display">Page title</h1>
        ...
    </section>
    <section class="oc-section">
        <h2 class="oc-h">Section heading</h2>   <!-- "## " prefix is added by CSS -->
        <hr class="oc-rule">
        <p class="oc-body oc-max">Body copy.</p>
    </section>
</div>

{% include 'components/_footer.html' %}
```

Pages currently on the system: `pages/index.html`, `pages/projects.html`,
`pages/getting_started.html`, `pages/about.html`, `pages/benchmarks.html`,
`pages/contributing.html` (docs pages keep their hand-written SEO `<head>` and
include `_docs_navbar.html` for the breadcrumb strip).

The footer is pinned to the bottom of the viewport on short pages: `oc.css`
makes `body.landing-page` a flex column and gives `> footer` an auto top
margin, so a 404 or a stub page no longer leaves it floating mid-screen. It
carries the copyright line plus the Privacy Policy and Terms of Service links.

## Palette (CSS variables on `:root`)

Blue family keyed to the logo badge (`#1049a2`). Never hardcode colors in page
markup; use the variables.

| Variable | Value | Role |
|----------|-------|------|
| `--canvas` | `#eef4fa` | Page background |
| `--soft` | `#e3edf7` | Subtle fill (inline code, hover) |
| `--card` | `#d7e4f2` | Raised surface (terminals, snippets) |
| `--ink` | `#0a2245` | Headings, primary buttons |
| `--body` | `#33517a` | Body text |
| `--mute` | `#5f7699` | Captions, secondary text |
| `--hairline` | `rgba(10,34,69,0.13)` | Default border |
| `--hairline-strong` | `rgba(10,34,69,0.38)` | Emphasized border / hover |
| `--accent` | `#2563c4` | Links, active elements |
| `--accent-deep` | `#1049a2` | Brand blue, tool names, code keywords |
| `--green` | `#0d9488` | Success, `[open]` status, prompt `>` |
| `--orange` | `#c2620c` | Numbers, warnings, `[in progress]` |
| `--dark` / `--dark-elev` / `--on-dark*` | — | Dark-surface variants (unused on light pages, kept for dark blocks) |

## Type

One face everywhere: JetBrains Mono. Roles by class, not by element:

- `.oc-display` — 38px/700 page title (28px under 640px)
- `.oc-h` — 16px/700 section heading; CSS prepends a muted `## `
- `.oc-body` — 16px body; `strong` renders in ink
- `.oc-caption` — 14px muted
- `.oc-sub` — 14px/700 ink in-section subhead (docs pages), no `##` prefix
- `.oc-max` — caps a text block at 720px
- `.oc-rule` — 1px hairline `<hr>` under section headings
- Inline links in running text (`.oc-body`, `.oc-caption`, `.why-text`,
  `.list-row`) render ink and underlined, brand blue on hover. Buttons and the
  TOC rail keep their own treatment.

## Layout

- `.oc-wrap` — the single 960px column, hairline side rails
- `.oc-section` — 96px vertical padding, hairline top rule; every section reads
  as an outlined box. First section on a page usually overrides to
  `padding-top: 64px` since the nav already provides space.
- Several `.oc-h` headings may share ONE section when each block is a short
  piece of running text (the privacy and terms pages: a section box per clause
  is mostly padding). CSS gives every heading after the first 56px of air;
  hang the `#anchor` id on the `<h2>` so the TOC rail still works.

## Components

- **Buttons** `.btn` + `.btn-primary` (ink) / `.btn-secondary` (outlined).
  Inline-flex with an 8px gap; put a 16px `currentColor` SVG icon before the
  label. No icon fonts.
- **Tabs** `.oc-tabs` strip of `.oc-tab` buttons + `.oc-panel` blocks.
  Auto-wired by oc.js: wrap in an element with `data-tab-group`, give each tab
  `data-tab="<id>"` and each panel `data-panel="<id>"`; mark one of each
  `.active`.
- **Snippet** `.snippet` one-line command pill with `.dollar`, `.cmd`, and a
  `.copy-btn`.
- **Code block** `.code-block` pre-formatted card surface. Syntax spans:
  `.kw` keyword, `.fn` function, `.str` string/success, `.cm` comment/muted,
  `.num` number, `.key` key/tool name. oc.js appends a `[copy]` button to every
  `.code-block` automatically.
- **Scroll-reveal terminal** add `.term-reveal` to a `.code-block`, wrap each
  line in `<div class="sl">…</div>` (blank lines: `&nbsp;`). oc.js reveals the
  lines one by one the first time the block scrolls into view; default stagger
  110ms, override per line with `data-d="450"`. Reduced motion gets the
  finished frame.
- **List rows** `.list-row` with `.m` bracket marker, `.lbl` bold label,
  `.txt` description. Two-column via `.list-grid` (collapses under 851px), or
  single column capped at the body measure via `.list-col` when each row's
  description is a full sentence. Add `.ok` or `.no` to the marker for a green
  affirmative (`[x]`) or an orange caution (`[!]`): do/don't lists, checklists
  and inline notes are the same component, only the bracket color changes.
- **Why cards** `.why-grid` of `.why-card` (`.why-num`, `.why-title`,
  `.why-text`) — 3-up principle tiles.
- **Figure tiles** `.fig-grid` of `.fig-tile` (`.fig-spark`, `.fig-num`,
  `.fig-cap`) — stat strips.
- **FAQ** `.faq-item` `<details>` rows with `.faq-summary` / `.faq-body`.
- **Board** (projects page): `.oc-search` input, `.oc-pill` filter chips,
  `.pcard-grid` of `.pcard` cards (`.pcard-top/-cat/-title/-sub/-desc/-tags/
  -tag/-foot`), status colors `.st-open/.st-testing/.st-progress/.st-planned`,
  difficulty colors `.df-first/.df-inter/.df-adv`, `.oc-skel` loading shimmer.
- **Marquee cards** (landing): `.bb-mq-wrap` / `.bb-track` / `.bb-card`, sync
  line `.bb-sync` (add `.stale` when showing fallback data).
- **Data tables** (benchmarks page): `.oc-table-wrap` around an `.oc-table`.
  First column left-aligned ink, other cells centered; helper cells `.size`
  (right-aligned muted), `.std` (± noise), `.best` (green best-on-dataset
  highlight), `.na`, `.tmo` (timeout). Charts keep the benchmark JSON's
  framework colors (`#f97316` TuiML / `#3b82f6` sklearn / `#a855f7` Weka);
  set Chart.js `defaults.font.family` to the mono face on chart pages.
- **Person cards** (about page): `.person-list` of `.person` rows, each a
  photo + `.person-name` / `.person-role` / `.person-org` / `.person-bio`.
  Photo stacks above the text under 640px.
- **Sidebar rail** (docs pages): THE one sidebar component, used by
  getting-started ("On this page") and benchmarks (algorithm selector).
  Markup: `<nav class="oc-toc oc-toc-flow">` with one or more `.oc-toc-label`
  group labels followed by entries, placed inside the section it belongs to.
  Entries get a `[+]` bracket marker; add class `sub` for an indented `[-]`
  subsection entry. Entries are `#anchor` links (oc.js scrollspies them,
  marks the current one `.active`, and keeps a sub's parent section
  highlighted) or `<button>`s for interactive selectors (page JS manages
  `.active`); plain links (e.g. a download) also work. Anchor targets scroll
  smoothly and clear the sticky nav via a global `scroll-margin-top`. On viewports ≥ 1400px
  it renders as a fixed rail left of the column; below that, `.oc-toc-flow`
  keeps it as an in-flow block where it stands. Omit `.oc-toc-flow` only if
  the rail is pure navigation that can safely disappear on narrow screens.

## Rules of the language

1. **Brackets are the icons.** Markers like `[+]`, `[01]`, `[open]`,
   `[needs testing]` replace icon fonts. The only SVGs are button icons and
   third-party logo marks.
2. **Flat on canvas.** No shadows, no gradients, no translate-on-hover. Hover
   feedback is a border or color change.
3. **4px radius, hairline borders** on every surface.
4. **Status colors**: green = open/success, brand blue = active/testing,
   orange = numbers/in-progress, mute = planned/secondary.
5. **Respect reduced motion.** Every animation (hero session, marquee,
   term-reveal, shimmer) has a `prefers-reduced-motion` fallback.
6. **No em dashes in section copy**; use colons or commas.

## JS helpers (`window.OC`)

- `OC.esc(s)` — HTML-escape untrusted text
- `OC.mdLite(s)` — escape, then render `` `code` `` and `**bold**` only; safe
  for remote text like GitHub issue bodies

## Adding a new page

1. Copy the skeleton above; pick `active_nav`.
2. Build sections from existing components before inventing new CSS.
3. If a new component is genuinely reusable, add it to `oc.css` (and its
   behavior to `oc.js`), document it here, and keep page files free of large
   `<style>` blocks.
