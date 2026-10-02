window.MathJax = {
  tex: {inlineMath: [["\\(", "\\)"]], displayMath: [["\\[", "\\]"]]},
  options: {ignoreHtmlClass: ".*|", processHtmlClass: "arithmatex"}
};
document$.subscribe(() => {
  if (window.MathJax.typesetPromise) {
    window.MathJax.typesetClear();
    window.MathJax.typesetPromise();
  }
});
