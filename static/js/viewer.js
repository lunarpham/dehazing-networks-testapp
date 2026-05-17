/**
 * SyncedViewer — Side-by-side image comparison with synced pan/zoom.
 *
 * Both canvases share a single transform state so every zoom and pan
 * action on one side is mirrored exactly on the other.
 */

class SyncedViewer {
    constructor() {
        // Canvas elements
        this.canvasLeft = document.getElementById('canvas-left');
        this.canvasRight = document.getElementById('canvas-right');
        this.ctxLeft = this.canvasLeft.getContext('2d');
        this.ctxRight = this.canvasRight.getContext('2d');

        // Container / UI
        this.panelLeft = document.getElementById('panel-left');
        this.panelRight = document.getElementById('panel-right');
        this.viewerPanels = document.getElementById('viewer-panels');
        this.viewerEmpty = document.getElementById('viewer-empty-state');
        this.zoomIndicator = document.getElementById('zoom-indicator');

        // Images
        this.imgLeft = null;
        this.imgRight = null;

        // Shared transform state
        this.transform = {
            scale: 1,
            offsetX: 0,
            offsetY: 0,
        };

        // Interaction state
        this._dragging = false;
        this._dragStartX = 0;
        this._dragStartY = 0;
        this._dragStartOffsetX = 0;
        this._dragStartOffsetY = 0;

        // Zoom limits
        this.MIN_SCALE = 0.1;
        this.MAX_SCALE = 32;
        this.ZOOM_FACTOR = 1.12;

        // Animation frame tracking
        this._rafId = null;
        this._needsRender = false;

        this._initEvents();
        this._resizeCanvases();

        // Handle window resize
        this._resizeObserver = new ResizeObserver(() => {
            this._resizeCanvases();
            this._requestRender();
        });
        this._resizeObserver.observe(this.panelLeft);
        this._resizeObserver.observe(this.panelRight);
    }

    // ── Public API ──────────────────────────────────────────────────────

    /**
     * Load images into the viewer.
     * @param {string} leftSrc - URL for the left (original) image
     * @param {string} rightSrc - URL for the right (output) image
     */
    loadImages(leftSrc, rightSrc) {
        let loaded = 0;
        const onLoad = () => {
            loaded++;
            if (loaded === 2) {
                this._show();
                this.fitToView();
            }
        };

        this.imgLeft = new Image();
        this.imgLeft.onload = onLoad;
        this.imgLeft.src = leftSrc;

        this.imgRight = new Image();
        this.imgRight.onload = onLoad;
        this.imgRight.src = rightSrc;
    }

    /**
     * Reset transform so the image fits in the viewport.
     */
    fitToView() {
        if (!this.imgLeft) return;

        const cw = this.canvasLeft.width;
        const ch = this.canvasLeft.height;
        const iw = this.imgLeft.naturalWidth;
        const ih = this.imgLeft.naturalHeight;

        const scaleX = cw / iw;
        const scaleY = ch / ih;
        this.transform.scale = Math.max(scaleX, scaleY);

        // Center the image
        this.transform.offsetX = (cw - iw * this.transform.scale) / 2;
        this.transform.offsetY = (ch - ih * this.transform.scale) / 2;

        this._updateZoomIndicator();
        this._requestRender();
    }

    /**
     * Set zoom to 100% (1:1 pixel mapping), centered.
     */
    zoomActualSize() {
        if (!this.imgLeft) return;

        const cw = this.canvasLeft.width;
        const ch = this.canvasLeft.height;
        const iw = this.imgLeft.naturalWidth;
        const ih = this.imgLeft.naturalHeight;

        this.transform.scale = 1;
        this.transform.offsetX = (cw - iw) / 2;
        this.transform.offsetY = (ch - ih) / 2;

        this._updateZoomIndicator();
        this._requestRender();
    }

    // ── Private: Rendering ──────────────────────────────────────────────

    _show() {
        this.viewerEmpty.style.display = 'none';
        this.viewerPanels.style.display = 'flex';
    }

    _resizeCanvases() {
        // Size canvases to match their CSS dimensions at device pixel ratio
        const dpr = window.devicePixelRatio || 1;

        for (const panel of [this.panelLeft, this.panelRight]) {
            const canvas = panel.querySelector('canvas');
            const rect = panel.getBoundingClientRect();
            canvas.width = rect.width * dpr;
            canvas.height = rect.height * dpr;
            canvas.getContext('2d').setTransform(dpr, 0, 0, dpr, 0, 0);
        }
    }

    _requestRender() {
        if (!this._needsRender) {
            this._needsRender = true;
            this._rafId = requestAnimationFrame(() => {
                this._render();
                this._needsRender = false;
            });
        }
    }

    _render() {
        this._drawCanvas(this.ctxLeft, this.canvasLeft, this.imgLeft);
        this._drawCanvas(this.ctxRight, this.canvasRight, this.imgRight);
    }

    _drawCanvas(ctx, canvas, img) {
        const dpr = window.devicePixelRatio || 1;
        const w = canvas.width / dpr;
        const h = canvas.height / dpr;

        // Clear with dark background
        ctx.fillStyle = '#060a13';
        ctx.fillRect(0, 0, w, h);

        if (!img || !img.complete) return;

        const { scale, offsetX, offsetY } = this.transform;
        const iw = img.naturalWidth;
        const ih = img.naturalHeight;

        // Draw checkerboard under the image area for context
        this._drawCheckerboard(ctx, offsetX, offsetY, iw * scale, ih * scale);

        // Enable crisp rendering for pixel-level zoom
        ctx.imageSmoothingEnabled = scale < 4;
        ctx.imageSmoothingQuality = 'high';

        // Draw image
        ctx.drawImage(img, offsetX, offsetY, iw * scale, ih * scale);

        // Draw border around image
        ctx.strokeStyle = 'rgba(148, 163, 184, 0.1)';
        ctx.lineWidth = 1;
        ctx.strokeRect(offsetX, offsetY, iw * scale, ih * scale);
    }

    _drawCheckerboard(ctx, x, y, w, h) {
        const size = 8;
        ctx.save();
        ctx.beginPath();
        ctx.rect(x, y, w, h);
        ctx.clip();

        const startCol = Math.floor(x / size);
        const endCol = Math.ceil((x + w) / size);
        const startRow = Math.floor(y / size);
        const endRow = Math.ceil((y + h) / size);

        for (let row = startRow; row < endRow; row++) {
            for (let col = startCol; col < endCol; col++) {
                ctx.fillStyle = (row + col) % 2 === 0 ? '#0d1420' : '#111827';
                ctx.fillRect(col * size, row * size, size, size);
            }
        }
        ctx.restore();
    }

    _updateZoomIndicator() {
        const pct = Math.round(this.transform.scale * 100);
        this.zoomIndicator.textContent = pct + '%';
    }

    // ── Private: Event Handling ─────────────────────────────────────────

    _initEvents() {
        // Wheel zoom on both panels
        for (const panel of [this.panelLeft, this.panelRight]) {
            panel.addEventListener('wheel', (e) => this._onWheel(e, panel), { passive: false });
            panel.addEventListener('mousedown', (e) => this._onMouseDown(e));
        }

        // Mouse move/up on document so dragging works outside the canvas
        document.addEventListener('mousemove', (e) => this._onMouseMove(e));
        document.addEventListener('mouseup', (e) => this._onMouseUp(e));

        // Toolbar buttons
        document.getElementById('btn-zoom-fit').addEventListener('click', () => this.fitToView());
        document.getElementById('btn-zoom-100').addEventListener('click', () => this.zoomActualSize());
    }

    _onWheel(e, panel) {
        e.preventDefault();
        if (!this.imgLeft) return;

        const rect = panel.getBoundingClientRect();

        // Mouse position relative to the canvas (CSS pixels)
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;

        // Zoom direction
        const delta = e.deltaY < 0 ? this.ZOOM_FACTOR : 1 / this.ZOOM_FACTOR;
        const newScale = Math.min(Math.max(this.transform.scale * delta, this.MIN_SCALE), this.MAX_SCALE);
        const ratio = newScale / this.transform.scale;

        // Zoom toward the cursor position
        this.transform.offsetX = mx - (mx - this.transform.offsetX) * ratio;
        this.transform.offsetY = my - (my - this.transform.offsetY) * ratio;
        this.transform.scale = newScale;

        this._updateZoomIndicator();
        this._requestRender();
    }

    _onMouseDown(e) {
        if (e.button !== 0) return;
        this._dragging = true;
        this._dragStartX = e.clientX;
        this._dragStartY = e.clientY;
        this._dragStartOffsetX = this.transform.offsetX;
        this._dragStartOffsetY = this.transform.offsetY;
    }

    _onMouseMove(e) {
        if (!this._dragging) return;

        const dx = e.clientX - this._dragStartX;
        const dy = e.clientY - this._dragStartY;

        this.transform.offsetX = this._dragStartOffsetX + dx;
        this.transform.offsetY = this._dragStartOffsetY + dy;

        this._requestRender();
    }

    _onMouseUp(e) {
        this._dragging = false;
    }
}

// Expose globally
window.SyncedViewer = SyncedViewer;
