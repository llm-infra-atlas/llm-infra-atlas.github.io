window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  },
  startup: {
    // The initial page goes through the same document$ path as instant
    // navigations, so MathJax's own one-shot initial typeset is skipped
    // (otherwise the first page would be typeset twice).
    typeset: false
  }
};

// Material emits document$ on the initial page and on every instant
// navigation. Serialize the typesets: overlapping typesetPromise calls render
// the same MathJax document concurrently and corrupt each other. All MathJax
// APIs used below exist synchronously once tex-mml-chtml.js has executed,
// which is guaranteed before the first document$ emission (script order plus
// DOMContentLoaded), so no readiness gate is needed.
let typesetQueue = Promise.resolve();
document$.subscribe(() => {
  typesetQueue = typesetQueue.then(() => {
    const MathJax = window.MathJax;
    // Instant navigation rebuilds <head>, orphaning the <style> element cached
    // by the CHTML output jax: its sheet turns null and every insertRule
    // silently fails, leaving formulas unstyled. clearCache() makes MathJax
    // recreate the stylesheet in the current head; texReset() keeps equation
    // numbering per page. (Official instant-navigation recipe.)
    MathJax.startup.output.clearCache();
    MathJax.typesetClear();
    MathJax.texReset();
    return MathJax.typesetPromise();
  }).catch(error => {
    // A failed page must not poison the queue for the next navigation.
    console.warn("MathJax typesetting failed", error);
  });
});
