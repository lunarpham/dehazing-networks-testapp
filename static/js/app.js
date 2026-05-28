/**
 * DehazeTest — Main Application Logic
 *
 * Handles model upload/management, image upload, inference, and
 * connecting everything to the SyncedViewer.
 */

(function () {
    'use strict';

    // ── State ───────────────────────────────────────────────────────────

    let selectedModelId = null;
    let selectedImageFile = null;
    let viewer = null;
    let currentModels = [];

    // ── DOM References ──────────────────────────────────────────────────

    const modelDropZone = document.getElementById('model-drop-zone');
    const modelFileInput = document.getElementById('model-file-input');
    const modelUploadForm = document.getElementById('model-upload-form');
    const selectedModelFile = document.getElementById('selected-model-file');
    const btnRemoveModelFile = document.getElementById('btn-remove-model-file');
    const archSelect = document.getElementById('arch-select');
    const btnUploadModel = document.getElementById('btn-upload-model');
    const modelList = document.getElementById('model-list');
    const noModelsMsg = document.getElementById('no-models-msg');

    const imageDropZone = document.getElementById('image-drop-zone');
    const imageFileInput = document.getElementById('image-file-input');
    const imagePreviewContainer = document.getElementById('image-preview-container');
    const imagePreview = document.getElementById('image-preview');
    const btnRemoveImage = document.getElementById('btn-remove-image');
    const btnRunInference = document.getElementById('btn-run-inference');

    const viewerEmptyState = document.getElementById('viewer-empty-state');
    const viewerPanels = document.getElementById('viewer-panels');
    const btnNewImage = document.getElementById('btn-new-image');

    const modelInfoBadge = document.getElementById('model-info-badge');
    const modelInfoName = document.getElementById('model-info-name');
    const modelInfoArch = document.getElementById('model-info-arch');

    const deviceBadge = document.getElementById('device-badge');

    // ── Initialize ──────────────────────────────────────────────────────

    function init() {
        viewer = new SyncedViewer();
        setupDropZone(modelDropZone, modelFileInput, handleModelFileSelected);
        setupDropZone(imageDropZone, imageFileInput, handleImageFileSelected);
        setupModelUpload();
        setupImageUpload();
        loadModelList();
        detectDevice();
    }

    // ── Device Detection ────────────────────────────────────────────────

    function detectDevice() {
        // We just show that the app is connected
        deviceBadge.textContent = 'Connected';
        deviceBadge.classList.add('badge-success');
    }

    // ── Toast Notifications ─────────────────────────────────────────────

    function showToast(message, type = 'info', duration = 4000) {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast toast-${type} fade-in`;
        toast.textContent = message;
        container.appendChild(toast);

        setTimeout(() => {
            toast.classList.add('toast-out');
            toast.addEventListener('animationend', () => toast.remove());
        }, duration);
    }

    // ── Drag & Drop Helper ──────────────────────────────────────────────

    function setupDropZone(zone, input, onFile) {
        zone.addEventListener('click', () => input.click());

        zone.addEventListener('dragover', (e) => {
            e.preventDefault();
            zone.classList.add('drag-over');
        });

        zone.addEventListener('dragleave', () => {
            zone.classList.remove('drag-over');
        });

        zone.addEventListener('drop', (e) => {
            e.preventDefault();
            zone.classList.remove('drag-over');
            if (e.dataTransfer.files.length > 0) {
                onFile(e.dataTransfer.files[0]);
            }
        });

        input.addEventListener('change', () => {
            if (input.files.length > 0) {
                onFile(input.files[0]);
            }
        });
    }

    // ── Model Upload ────────────────────────────────────────────────────

    let pendingModelFile = null;

    function handleModelFileSelected(file) {
        if (!file.name.endsWith('.pth')) {
            showToast('Only .pth files are accepted', 'error');
            return;
        }
        pendingModelFile = file;
        selectedModelFile.querySelector('.file-name').textContent = file.name;
        modelUploadForm.style.display = 'flex';
        modelDropZone.style.display = 'none';
    }

    function setupModelUpload() {
        btnRemoveModelFile.addEventListener('click', () => {
            pendingModelFile = null;
            modelFileInput.value = '';
            modelUploadForm.style.display = 'none';
            modelDropZone.style.display = 'block';
        });

        btnUploadModel.addEventListener('click', async () => {
            if (!pendingModelFile) return;

            btnUploadModel.disabled = true;
            btnUploadModel.textContent = 'Uploading…';

            const formData = new FormData();
            formData.append('file', pendingModelFile);
            formData.append('arch_type', archSelect.value);

            try {
                const res = await fetch('/api/models/upload', {
                    method: 'POST',
                    body: formData,
                });
                const data = await res.json();

                if (!res.ok) {
                    showToast(data.error || 'Upload failed', 'error');
                    return;
                }

                showToast(`Model "${data.name}" uploaded successfully`, 'success');
                pendingModelFile = null;
                modelFileInput.value = '';
                modelUploadForm.style.display = 'none';
                modelDropZone.style.display = 'block';
                loadModelList();
            } catch (err) {
                showToast('Upload failed: ' + err.message, 'error');
            } finally {
                btnUploadModel.disabled = false;
                btnUploadModel.innerHTML = `
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                    Upload Model
                `;
            }
        });
    }

    // ── Model List ──────────────────────────────────────────────────────

    async function loadModelList() {
        try {
            const res = await fetch('/api/models');
            const data = await res.json();
            renderModelList(data.models);
        } catch (err) {
            console.error('Failed to load models:', err);
        }
    }

    // Map arch type to human-readable names
    const archNames = {
        'dehazenet': 'DehazeNet',
        'aodnet': 'AOD-Net',
        'msfa_denet': 'MSFA-DeNet',
    };

    function renderModelList(models) {
        currentModels = models;

        // Remove all model items (keep the empty-state div)
        const items = modelList.querySelectorAll('.model-item');
        items.forEach(item => item.remove());

        if (models.length === 0) {
            noModelsMsg.style.display = 'block';
            return;
        }

        noModelsMsg.style.display = 'none';

        models.forEach((model) => {
            const item = document.createElement('div');
            item.className = 'model-item fade-in';
            if (model.id === selectedModelId) {
                item.classList.add('selected');
            }
            item.dataset.id = model.id;

            item.innerHTML = `
                <div class="model-item-radio"></div>
                <div class="model-item-info">
                    <div class="model-item-name" title="${model.name}">${model.name}</div>
                    <div class="model-item-arch">${archNames[model.arch_type] || model.arch_type}</div>
                </div>
                <button class="btn-icon-run btn-run-model" title="Run inference with this model" data-id="${model.id}">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polygon points="5 3 19 12 5 21 5 3"/>
                    </svg>
                </button>
                <button class="btn-icon btn-delete-model" title="Delete model" data-id="${model.id}">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="3 6 5 6 21 6"/>
                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                    </svg>
                </button>
            `;

            // Click to select
            item.addEventListener('click', (e) => {
                if (e.target.closest('.btn-delete-model') || e.target.closest('.btn-run-model')) return;
                selectedModelId = model.id;
                renderModelList(models);
                updateInferenceButton();
            });

            // Run button
            item.querySelector('.btn-run-model').addEventListener('click', async (e) => {
                e.stopPropagation();
                selectedModelId = model.id;
                renderModelList(models);
                runInference(model.id);
            });

            // Delete button
            item.querySelector('.btn-delete-model').addEventListener('click', async (e) => {
                e.stopPropagation();
                await deleteModel(model.id);
            });

            modelList.appendChild(item);
        });

        updateInferenceButton();
    }

    async function deleteModel(modelId) {
        try {
            const res = await fetch(`/api/models/${modelId}`, { method: 'DELETE' });
            const data = await res.json();

            if (!res.ok) {
                showToast(data.error || 'Delete failed', 'error');
                return;
            }

            if (selectedModelId === modelId) {
                selectedModelId = null;
            }

            showToast('Model deleted', 'info');
            loadModelList();
        } catch (err) {
            showToast('Delete failed: ' + err.message, 'error');
        }
    }

    // ── Image Upload ────────────────────────────────────────────────────

    function handleImageFileSelected(file) {
        if (!file.type.startsWith('image/')) {
            showToast('Please select an image file', 'error');
            return;
        }
        selectedImageFile = file;

        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreviewContainer.style.display = 'block';
            imageDropZone.style.display = 'none';
            updateInferenceButton();
        };
        reader.readAsDataURL(file);
    }

    function resetToUpload() {
        selectedImageFile = null;
        imageFileInput.value = '';
        imagePreviewContainer.style.display = 'none';
        imageDropZone.style.display = 'block';

        // Reset viewer back to upload state
        viewerPanels.style.display = 'none';
        viewerEmptyState.style.display = 'flex';
        btnNewImage.style.display = 'none';

        hideModelInfo();
        updateInferenceButton();
    }

    function setupImageUpload() {
        btnRemoveImage.addEventListener('click', resetToUpload);
        btnNewImage.addEventListener('click', resetToUpload);
        btnRunInference.addEventListener('click', runInference);
    }

    // ── Inference ───────────────────────────────────────────────────────

    function updateInferenceButton() {
        btnRunInference.disabled = !(selectedModelId && selectedImageFile);

        // Also update run buttons in model list
        const runBtns = modelList.querySelectorAll('.btn-run-model');
        runBtns.forEach(btn => {
            btn.disabled = !selectedImageFile;
        });
    }

    function showModelInfo(modelId) {
        const model = currentModels.find(m => m.id === modelId);
        if (model && modelInfoBadge) {
            modelInfoName.textContent = model.name;
            modelInfoArch.textContent = archNames[model.arch_type] || model.arch_type;
            modelInfoBadge.style.display = 'inline-flex';
        }
    }

    function hideModelInfo() {
        if (modelInfoBadge) {
            modelInfoBadge.style.display = 'none';
        }
    }

    async function runInference(overrideModelId) {
        const modelId = overrideModelId || selectedModelId;
        if (!modelId || !selectedImageFile) {
            if (!selectedImageFile) {
                showToast('Please upload an image first', 'error');
            }
            return;
        }

        // Show loading state
        const btnText = btnRunInference.querySelector('.btn-text');
        const btnSpinner = btnRunInference.querySelector('.btn-spinner');
        btnText.style.display = 'none';
        btnSpinner.style.display = 'inline-flex';
        btnRunInference.disabled = true;

        // Show spinner on the model-item run button if clicked from there
        const runBtn = modelList.querySelector(`.btn-run-model[data-id="${modelId}"]`);
        if (runBtn) {
            runBtn.classList.add('running');
            runBtn.disabled = true;
        }

        const formData = new FormData();
        formData.append('image', selectedImageFile);
        formData.append('model_id', modelId);

        try {
            const res = await fetch('/api/infer', {
                method: 'POST',
                body: formData,
            });
            const data = await res.json();

            if (!res.ok) {
                showToast(data.error || 'Inference failed', 'error');
                return;
            }

            // Hide the upload card, show comparison panels
            viewerEmptyState.style.display = 'none';
            viewerPanels.style.display = 'flex';
            btnNewImage.style.display = 'inline-flex';

            // Load both images into the synced viewer
            viewer.loadImages(data.input_url, data.output_url);

            // Show which model was used
            showModelInfo(modelId);

            showToast('Inference complete!', 'success');

        } catch (err) {
            showToast('Inference failed: ' + err.message, 'error');
        } finally {
            btnText.style.display = 'inline';
            btnSpinner.style.display = 'none';
            btnRunInference.disabled = false;
            if (runBtn) {
                runBtn.classList.remove('running');
                runBtn.disabled = false;
            }
            updateInferenceButton();
        }
    }

    // ── Boot ────────────────────────────────────────────────────────────

    document.addEventListener('DOMContentLoaded', init);

})();
