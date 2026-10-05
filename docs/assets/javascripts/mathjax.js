window.MathJax = {
  tex: {inlineMath: [["\\(", "\\)"]], displayMath: [["\\[", "\\]"]]},
  options: {ignoreHtmlClass: ".*|", processHtmlClass: "arithmatex"},
  // Instant navigation removes dynamic styles; include all glyph rules up front.
  chtml: {adaptiveCSS: false},
  startup: {
    input: ["tex"],
    typeset: false,
    pageReady() {
      let pending = Promise.resolve();
      let previousArticle;
      document$.subscribe(() => {
        const article = document.querySelector("article");
        if (!article || article === previousArticle) return;
        previousArticle = article;
        pending = pending.then(() => {
          if (!article.isConnected) return;
          MathJax.typesetClear();
          return MathJax.typesetPromise([article]);
        }).catch(error => console.error("MathJax typesetting failed:", error));
        MathJax.startup.promise = pending;
      });
      return pending;
    }
  }
};
