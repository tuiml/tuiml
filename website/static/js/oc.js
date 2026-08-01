/* ==================================================================
   TuiML site design system ("oc") — shared behaviors.
   Pair with /static/css/oc.css. See website/DESIGN.md.

   Auto-wired on DOM ready:
     1. Tab strips    — any .oc-tab with data-tab="<id>" toggles the
                        .oc-panel with a matching data-panel="<id>",
                        scoped to the nearest [data-tab-group].
     2. Copy buttons  — every .code-block gets a [copy] button that
                        copies its text (captured before the button is
                        appended, so "[copy]" never lands in the clipboard).
     3. Term reveal   — elements with class .term-reveal reveal their
                        .sl lines one by one the first time they scroll
                        into view. Per-line extra delay via data-d="ms".
                        Reduced motion / no IntersectionObserver get the
                        finished frame instantly.
   ================================================================== */
(function () {
    'use strict';

    /* ---------------- 1. Tab strips ---------------- */
    function wireTabs() {
        document.querySelectorAll('[data-tab-group]').forEach(group => {
            const tabs = group.querySelectorAll('.oc-tab[data-tab]');
            const panels = group.querySelectorAll('.oc-panel[data-panel]');
            tabs.forEach(tab => {
                tab.addEventListener('click', () => {
                    tabs.forEach(t => t.classList.toggle('active', t === tab));
                    panels.forEach(p => p.classList.toggle('active',
                        p.dataset.panel === tab.dataset.tab));
                });
            });
        });
    }

    /* ---------------- 2. Copy buttons on code blocks ---------------- */
    function wireCopyButtons() {
        document.querySelectorAll('.code-block').forEach(el => {
            if (el.querySelector('.code-copy')) return;
            const text = el.innerText.replace(/\s+$/, '');
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'code-copy';
            btn.textContent = '[copy]';
            btn.setAttribute('aria-label', 'Copy code');
            btn.addEventListener('click', () => {
                navigator.clipboard.writeText(text);
                btn.textContent = '[ok]';
                setTimeout(() => { btn.textContent = '[copy]'; }, 1500);
            });
            el.appendChild(btn);
        });
    }

    /* ---------------- 3. Scroll-reveal terminals ---------------- */
    function wireTermReveal() {
        const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        document.querySelectorAll('.term-reveal').forEach(term => {
            const lines = term.querySelectorAll('.sl');
            if (reduced || !('IntersectionObserver' in window)) {
                lines.forEach(l => l.classList.add('show'));
                return;
            }
            const io = new IntersectionObserver(entries => {
                if (!entries.some(e => e.isIntersecting)) return;
                io.disconnect();
                let t = 0;
                lines.forEach(l => {
                    t += parseInt(l.dataset.d || '110', 10);
                    setTimeout(() => l.classList.add('show'), t);
                });
            }, { threshold: 0.3 });
            io.observe(term);
        });
    }

    /* ---------------- 4. On-this-page TOC scrollspy ---------------- */
    function wireToc() {
        const toc = document.querySelector('.oc-toc');
        if (!toc) return;
        const pairs = Array.from(toc.querySelectorAll('a[href^="#"]'))
            .map(l => ({ l, el: document.getElementById(decodeURIComponent(l.hash.slice(1))) }))
            .filter(p => p.el);
        if (!pairs.length) return;

        /* Scroll via scrollIntoView instead of native fragment navigation:
           Chrome intermittently drops the smooth fragment scroll, leaving the
           hash updated but the page unmoved. scrollIntoView honors the
           targets' scroll-margin-top, so headings clear the sticky nav. */
        const smooth = window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth';
        pairs.forEach(p => {
            p.l.addEventListener('click', e => {
                e.preventDefault();
                p.el.scrollIntoView({ behavior: smooth });
                history.pushState(null, '', '#' + p.el.id);
            });
        });

        function update() {
            const pos = window.scrollY + 200;
            let idx = 0;
            pairs.forEach((p, i) => { if (pos >= p.el.offsetTop) idx = i; });
            pairs.forEach(p => p.l.classList.remove('active'));
            pairs[idx].l.classList.add('active');
            /* A subsection keeps its parent section highlighted too. */
            if (pairs[idx].l.classList.contains('sub')) {
                for (let i = idx; i >= 0; i--) {
                    if (!pairs[i].l.classList.contains('sub')) {
                        pairs[i].l.classList.add('active');
                        break;
                    }
                }
            }
        }
        window.addEventListener('scroll', update, { passive: true });
        update();
    }

    function init() {
        wireTabs();
        wireCopyButtons();
        wireTermReveal();
        wireToc();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    /* Escape/format helpers shared by the board pages. */
    window.OC = {
        esc: s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])),
        /* Minimal inline-markdown: `code` and **bold**. Escaped first, so
           remote text (e.g. GitHub issue bodies) can't inject HTML. */
        mdLite: s => window.OC.esc(s)
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>'),
    };
})();
