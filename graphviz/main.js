'use strict';

(function adjustGithubRepositoryUrl(a) {
  if (!(a && location.host.match('^([^.]+)\.github\.io$'))) { return; }
  const ghuser = RegExp.$1;
  const url = new URL(a.href);
  const pathSplit = url.pathname.split('/');
  pathSplit[1] = ghuser;
  url.pathname = pathSplit.join('/');
  a.href = url.href;
})(document.querySelector('#github-link'));

(function (document) {
  const GraphvizOnline = globalThis.GraphvizOnline ?? (globalThis.GraphvizOnline = new EventTarget());

  const editorPaneElement = document.getElementById('editor-pane');
  const reviewElement = document.getElementById('review');
  const toggleBtn = document.getElementById('toggle-btn');
  const splitter = document.getElementById('splitter');

  // Splitter drag and double-click functionality

  splitter.addEventListener('dblclick', (e) => {
    document.documentElement.style.setProperty('--editor-width', `calc(50% - var(--splitter-width))`);
  });
  splitter.addEventListener('mousedown', (e) => {
    document.documentElement.classList.add('splitter-is-dragging');
    // overlay to prevent mouse pointer flickering when dragging the splitter to the left
    const overlayElement = editorPaneElement.appendChild(document.createElement('div'));
    Object.assign(overlayElement.style, { position: 'absolute', inset: 0, zIndex: 10 });
    document.addEventListener('mousemove', handleMouseMoveWhileDraggingSplitter);
    document.addEventListener('mouseup', () => {
      document.documentElement.classList.remove('splitter-is-dragging');
      overlayElement.remove();
      document.removeEventListener('mousemove', handleMouseMoveWhileDraggingSplitter);
      resizeSVG();
    }, { once: true });
    e.preventDefault();
  });

  function handleMouseMoveWhileDraggingSplitter(e) {
    const windowWidth = window.innerWidth;
    const newLeft = e.clientX;
    const percentage = (newLeft / windowWidth) * 100;

    // Limit the splitter to reasonable bounds (10% to 90%)
    if (percentage >= 10 && percentage <= 90) {
      document.documentElement.style.setProperty('--editor-width', `${percentage}%`);
      resizeSVG({ throttle: 100 });
    }
  }

  toggleBtn.addEventListener('click', () => {
    document.documentElement.classList.toggle('editor-collapsed')
  });

  let resizeSVGTimer = null;
  window.addEventListener('resize', resizeSVG.bind(null, { throttle: 100 }));
  editorPaneElement.addEventListener('transitionend', resizeSVG);

  let prevReviewRect;
  // a valid bounding client rect is available after <body> is rendered
  // (i.e. document.documentElement.classList.contains('ready'))
  setTimeout(() => {
    prevReviewRect = reviewElement.getBoundingClientRect();
  }, 0);

  function resizeSVG({ throttle } = {}) {
    if (throttle) {
      if (resizeSVGTimer) { return; }
      resizeSVGTimer = setTimeout(resizeSVG, 100);
      return;
    } else {
      clearTimeout(resizeSVGTimer);
    }
    resizeSVGTimer = null;
    const svg = document.querySelector('#review svg');
    if (svg) {
      const currRect = reviewElement.getBoundingClientRect();
      const dw = currRect.width - prevReviewRect.width;
      const dh = currRect.height - prevReviewRect.height;
      svgPanZoom(svg).resize().panBy({ x: dw / 2, y: dh / 2 });
      prevReviewRect = currRect;
    }
  }

  //http://stackoverflow.com/a/10372280/398634
  window.URL = window.URL || window.webkitURL;
  var el_stetus = document.getElementById("status"),
    t_stetus = -1,
    reviewer = document.getElementById("review"),
    scale = window.devicePixelRatio || 1,
    downloadBtn = document.getElementById("download"),
    editor = ace.edit("editor"),
    lastHD = -1,
    worker = null,
    parser = new DOMParser(),
    showError = null,
    formatEl = document.querySelector("#format select"),
    engineEl = document.querySelector("#engine select"),
    rawEl = document.querySelector("#raw input"),
    shareEl = document.querySelector("#share"),
    shareURLEl = document.querySelector("#shareurl"),
    errorEl = document.querySelector("#error");

  // if the editor is initally hidden in presentation mode,
  // we need to trigger a resize once
  const boundEditorResize = editor.resize.bind(editor, true);
  editorPaneElement.addEventListener('transitionstart', boundEditorResize, { once: true });
  editorPaneElement.addEventListener('transitionend', boundEditorResize, { once: true });

  function show_status(text, hide) {
    hide = hide || 0;
    clearTimeout(t_stetus);
    el_stetus.innerHTML = text;
    if (hide) {
      t_stetus = setTimeout(function () {
        el_stetus.innerHTML = "";
      }, hide);
    }
  }

  function show_error(e) {
    console.trace();
    console.error(e);
    show_status("error", 500);
    reviewer.classList.remove("working");
    reviewer.classList.add("error");
    var message = e.message === undefined ? "An error occurred while processing the graph input." : e.message;
    while (errorEl.firstChild) {
      errorEl.removeChild(errorEl.firstChild);
    }
    errorEl.appendChild(document.createTextNode(message));
  }

  function svgXmlToImage(svgXml, callback) {
    var pngImage = new Image(), svgImage = new Image();

    svgImage.onload = function () {
      var canvas = document.createElement("canvas");
      canvas.width = svgImage.width * scale;
      canvas.height = svgImage.height * scale;

      var context = canvas.getContext("2d");
      context.drawImage(svgImage, 0, 0, canvas.width, canvas.height);

      pngImage.src = canvas.toDataURL("image/png");
      pngImage.width = svgImage.width;
      pngImage.height = svgImage.height;

      if (callback !== undefined) {
        callback(null, pngImage);
      }
    }

    svgImage.onerror = function (e) {
      if (callback !== undefined) {
        callback(e);
      }
    }
    svgImage.src = svgXml;
  }

  function copyShareURL(e) {
    let content = encodeURIComponent(editor.getSession().getDocument().getValue());
    const longUrl = new URL(location.href);
    longUrl.hash = content;
    // encode preserved commas from updateState()
    longUrl.search = longUrl.searchParams.toString();

    shareEl.disabled = true;
    let n = 0;
    let animateId = setInterval(()=> { shareEl.value = "Loading" + ".".repeat(n++%3)}, 300)
    // cors for is.gd
    fetch("https://api.allorigins.win/get?url=" + encodeURIComponent("https://is.gd/create.php?" + new URLSearchParams({
      format: 'simple',
      url: longUrl.toString(),
    }).toString()))
      .then((r) => {
        if (r.ok) return r.json()
        return new Error("network issues");
      })
      .then((rs) => {
        shareURLEl.style.display = "inline";
        shareURLEl.value = rs.contents;
      }).catch((err) => {
        const rawContent = editor.getSession().getDocument().getValue();
        const compressedContent = LZString.compressToEncodedURIComponent(rawContent);

        const compressedUrl = new URL(location.href);
        compressedUrl.searchParams.append("compressed", compressedContent);
        compressedUrl.hash = "";
        shareURLEl.style.display = "inline";
        shareURLEl.value = compressedUrl.toString();
      }).finally(()=>{
        clearInterval(animateId);
        shareEl.value = "Share";
        shareEl.disabled = false;
      })
  }

  function copyToClipboard(str) {
    const el = document.createElement('textarea');
    el.value = str;
    el.setAttribute('readonly', '');
    el.style.position = 'absolute';
    el.style.left = '-9999px';
    document.body.appendChild(el);
    const selected =
      document.getSelection().rangeCount > 0
        ? document.getSelection().getRangeAt(0)
        : false;
    el.select();
    var result = document.execCommand('copy')
    document.body.removeChild(el);
    if (selected) {
      document.getSelection().removeAllRanges();
      document.getSelection().addRange(selected);
    }
    return result;
  };

  function renderGraph() {
    reviewer.classList.add("working");
    reviewer.classList.remove("error");

    show_status("rendering...");
    Viz.instance().then(function (viz) {
      let dotContent = editor.getSession().getDocument().getValue();
      let options = {
        format: formatEl.value,
        engine: engineEl.value,
      };
      let result = null;

      if (["svg", "png"].indexOf(formatEl.value) > -1) {
        result = viz.renderSVGElement(dotContent, options);
      } else {
        result = viz.render(dotContent, options);
      }

      if (result.status !== undefined && result.status != "success") {
        show_error(result.errors && result.errors.length > 0 && result.errors[0] || result);
      } else {
        updateOutput(result);
      }
    }).catch((err) => {
      show_error(err);
    }).finally(() => {
      reviewer.classList.remove("working");
      show_status("done", 500)
    });
  }

  function updateState() {
    const updateStateEvent = new Event('updateState', { cancelable: true } );
    GraphvizOnline.dispatchEvent(updateStateEvent);
    if (updateStateEvent.defaultPrevented) { return; }

    const updatedUrl = new URL(window.location)
    // Hash
    const content = encodeURIComponent(editor.getSession().getDocument().getValue());
    updatedUrl.hash = content
    // Search params
    updatedUrl.searchParams.set("engine", engineEl.value);
    // preserve commas in search params
    updatedUrl.search = updatedUrl.searchParams.toString().replaceAll('%2C', ',');
    history.pushState({ "content": content, "engine": engineEl.value }, "", updatedUrl.toString())
  }

  function updateOutput(result) {
    const updateOutputEvent = new CustomEvent('updateOutput', { detail: { result } } );
    GraphvizOnline.dispatchEvent(updateOutputEvent);

    if (formatEl.value === "svg") {
      document.querySelector("#raw").classList.remove("disabled");
      rawEl.disabled = false;
    } else {
      document.querySelector("#raw").classList.add("disabled");
      rawEl.disabled = true;
    }

    var text = reviewer.querySelector("#text");
    if (text) {
      reviewer.removeChild(text);
    }

    var a = reviewer.querySelector("a");
    if (a) {
      reviewer.removeChild(a);
    }

    if (!result) {
      return;
    }

    reviewer.classList.remove("working");
    reviewer.classList.remove("error");

    if (formatEl.value == "svg" && !rawEl.checked) {
      var serializer = new XMLSerializer();
      var source = serializer.serializeToString(result);
      // https://stackoverflow.com/questions/18925210/download-blob-content-using-specified-charset
      //const blob = new Blob(["\ufeff", svg], {type: 'image/svg+xml;charset=utf-8'});
      const url = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(source);
      downloadBtn.href = url;
      downloadBtn.download = "graphviz.svg";
      var a = document.createElement("a");
      a.appendChild(result);
      reviewer.appendChild(a);
      svgPanZoom(result, {
        zoomEnabled: true,
        controlIconsEnabled: true,
        fit: true,
        center: true,
        // based on https://github.com/bumbu/svg-pan-zoom/blob/master/demo/mobile.html
        // added stuff around svgElement.getBoundingClientRect()
        customEventsHandler: {
          haltEventListeners: ['touchstart', 'touchend', 'touchmove', 'touchleave', 'touchcancel']
          , init: function (options) {
            var instance = options.instance
              , initialScale = 1
              , pannedX = 0
              , pannedY = 0
              , clientX = 0
              , clientY = 0
            // Init Hammer
            // Listen only for pointer and touch events
            this.hammer = Hammer(options.svgElement, {
              inputClass: Hammer.SUPPORT_POINTER_EVENTS ? Hammer.PointerEventInput : Hammer.TouchInput
            })
            // Enable pinch
            this.hammer.get('pinch').set({ enable: true })
            // Handle double tap
            this.hammer.on('doubletap', function (ev) {
              instance.zoomIn()
            })
            // Handle pan
            this.hammer.on('panstart panmove', function (ev) {
              // On pan start reset panned variables
              if (ev.type === 'panstart') {
                pannedX = 0
                pannedY = 0
              }
              // Pan only the difference
              instance.panBy({ x: ev.deltaX - pannedX, y: ev.deltaY - pannedY })
              pannedX = ev.deltaX
              pannedY = ev.deltaY
            })
            // Handle pinch
            this.hammer.on('pinchstart pinchmove', function (ev) {
              // On pinch start remember initial zoom
              if (ev.type === 'pinchstart') {
                ({x: clientX, y: clientY } = options.svgElement.getBoundingClientRect());
                initialScale = instance.getZoom()
              }
              instance.zoomAtPoint(initialScale * ev.scale, {
                x: ev.center.x - clientX,
                y: ev.center.y - clientY,
              });
            })
            // Prevent moving the page on some devices when panning over SVG
            options.svgElement.addEventListener('touchmove', function (e) { e.preventDefault(); });
          }
          , destroy: function () {
            this.hammer.destroy()
          }
        },
      });
    } else if (formatEl.value == "png") {
      var serializer = new XMLSerializer();
      var source = serializer.serializeToString(result);
      let resultWithPNGHeader = "data:image/svg+xml;base64," + btoa(unescape(encodeURIComponent(source)));
      svgXmlToImage(resultWithPNGHeader, function (err, image) {
        if (err) {
          show_error(err)
          return
        }
        image.setAttribute("title", "graphviz");
        downloadBtn.href = image.src;
        downloadBtn.download = "graphviz.png";
        var a = document.createElement("a");
        a.appendChild(image);
        reviewer.appendChild(a);
      })
    } else {
      var text = document.createElement("div");
      text.id = "text";
      if (formatEl.value == "svg") {
        let serializer = new XMLSerializer();
        result = serializer.serializeToString(result);
      } else {
        result = result.output;
      }
      text.appendChild(document.createTextNode(result));
      reviewer.appendChild(text);
    }

    updateState()
  }

  let renderGraphImmediately = false;
  editor.setTheme("ace/theme/twilight");
  editor.getSession().setMode("ace/mode/dot");
  editor.getSession().setValueAndRenderGraphImmediately = (value) => {
    renderGraphImmediately = true;
    editor.getSession().setValue(value);
  }
  editor.getSession().on("change", function (delta) {
    clearTimeout(lastHD);
    if (renderGraphImmediately) {
      lastHD = -1;
      // do not render after 'remove'
      if (delta.action === 'insert') {
        renderGraph();
        renderGraphImmediately = false;
      }
      return;
    }
    lastHD = setTimeout(renderGraph, 1500);
  });

  window.onpopstate = function (event) {
    if (event.state != null && event.state.content != undefined) {
      editor.getSession().setValue(decodeURIComponent(event.state.content));
    }
  };

  (function initEditorStableWidthGutterRenderer() {
    let numberOfLines;
    let gutterChars;
    let gutterWidth;
    let gutterPad;
    function ensureGutterPad(session) {
      const n = session.doc.$lines.length;
      if (n !== numberOfLines) {
        numberOfLines = n;
        gutterChars = String(n).length;
        gutterPad = ' '.repeat(gutterChars < 3 ? 3 : gutterChars);
        gutterWidth = undefined;
      }
    }
    function ensureGutterWidth(session, config) {
      ensureGutterPad(session);
      if (gutterWidth === undefined) {
        gutterWidth = gutterChars * config.characterWidth;
      }
      return gutterWidth;
    }
    editor.session.gutterRenderer = {
      getText: function (session, row) {
        ensureGutterPad(session);
        return (gutterPad + row).slice(-gutterChars);
      },
      getWidth: function (session, lastLineNumber, config) {
        ensureGutterWidth(session, config);
        return gutterWidth;
      },
    };
  }).call();

  formatEl.addEventListener("change", renderGraph);
  engineEl.addEventListener("change", renderGraph);
  rawEl.addEventListener("change", renderGraph);
  share.addEventListener("click", copyShareURL);

  // Since apparently HTMLCollection does not implement the oh so convenient array functions
  HTMLOptionsCollection.prototype.indexOf = function (name) {
    for (let i = 0; i < this.length; i++) {
      if (this[i].value == name) {
        return i;
      }
    }

    return -1;
  };

  Object.assign(GraphvizOnline, {
    editor,
    reviewElement,
    show_error,
  });
  const startupEvent = new Event('startup', { cancelable: true } );
  GraphvizOnline.dispatchEvent(startupEvent);
  if (startupEvent.defaultPrevented) { return; }

  /* parsing from URL sharing */
  const params = new URLSearchParams(location.search.substring(1));
  if (params.has('engine')) {
    const engine = params.get('engine');
    const index = engineEl.options.indexOf(engine);
    if (index > -1) { // if index exists
      engineEl.selectedIndex = index;
    } else {
      show_error({ message: `invalid engine ${engine} selected` });
    }
  }

  if (params.has('format')) {
    const format = params.get('format');
    const index = formatEl.options.indexOf(format);
    if (index > -1) {
      formatEl.selectedIndex = index;
    } else {
      show_error({ message: `Invalid format ${format} selected` });
    }
  }

  if (params.has('presentation')) {
    const classList = document.documentElement.classList;
    classList.add('editor-collapsed', 'hide-editor-toggle');
    const options = params.get('presentation').split(',');
    if (options.indexOf('editable') !== -1) {
      classList.remove('hide-editor-toggle');
    }
    for (const option of options) {
      if (option.startsWith('show-') || option.startsWith('hide-')) {
        classList.add(option);
      }
    }
  }

  if (params.has('raw')) {
    editor.getSession().setValueAndRenderGraphImmediately(params.get('raw'));
  } else if (params.has('compressed')) {
    const compressed = params.get('compressed');
    editor.getSession().setValueAndRenderGraphImmediately(LZString.decompressFromEncodedURIComponent(compressed));
  } else if (params.has('url')) {
    const url = params.get('url');
    let ok = false;
    fetch(url)
      .then(res => {
        ok = res.ok;
        return res.text();
      })
      .then(res => {
        if (!ok) {
          throw { message: res };
        }

        editor.getSession().setValueAndRenderGraphImmediately(res);
      }).catch(e => {
        show_error(e);
      });
  } else if (location.hash.length > 1) {
    editor.getSession().setValueAndRenderGraphImmediately(decodeURIComponent(location.hash.substring(1)));
  } else if (editor.getValue()) { // Init
    renderGraph();
  }

  document.documentElement.classList.add('ready');

})(document);
