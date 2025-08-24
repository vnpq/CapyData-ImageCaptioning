//  Config 
// const API_BASE_URL = "http://localhost:8000";
const API_BASE_URL = "http://0.0.0.0:8000";

//  State 
let currentImage = null;
let generatedTitles = [];

//  DOM 
// Desktop
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const imagePreview = document.getElementById('imagePreview');
const previewImg = document.getElementById('previewImg');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const imageDimensions = document.getElementById('imageDimensions');
const changeImageBtn = document.getElementById('changeImageBtn');
const removeImageBtn = document.getElementById('removeImageBtn');

// Mobile
const dropzoneMobile = document.getElementById('dropzoneMobile');
const fileInputMobile = document.getElementById('fileInputMobile');
const uploadBtnMobile = document.getElementById('uploadBtnMobile');
const imagePreviewMobile = document.getElementById('imagePreviewMobile');
const previewImgMobile = document.getElementById('previewImgMobile');
const fileNameMobile = document.getElementById('fileNameMobile');
const fileSizeMobile = document.getElementById('fileSizeMobile');
const imageDimensionsMobile = document.getElementById('imageDimensionsMobile');
const changeImageBtnMobile = document.getElementById('changeImageBtnMobile');
const removeImageBtnMobile = document.getElementById('removeImageBtnMobile');

// Shared
const generateBtn = document.getElementById('generateBtn');
const clearBtn = document.getElementById('clearBtn');
const titleCount = document.getElementById('titleCount');
const countBadge = document.getElementById('countBadge');
const statusRow = document.getElementById('statusRow');
const statusText = document.getElementById('statusText');
const loadingBar = document.getElementById('loadingBar');
const emptyState = document.getElementById('emptyState');
const loadingState = document.getElementById('loadingState');
const resultsList = document.getElementById('resultsList');
const errorState = document.getElementById('errorState');
const resultActions = document.getElementById('resultActions');
const copyAllBtn = document.getElementById('copyAllBtn');
const downloadBtn = document.getElementById('downloadBtn');
const toast = document.getElementById('toast');
const toastIcon = document.getElementById('toastIcon');
const toastMessage = document.getElementById('toastMessage');

//  Helpers 
const showResultsState = (state) => {
  emptyState.classList.add('hidden');
  loadingState.classList.add('hidden');
  resultsList.classList.add('hidden');
  errorState.classList.add('hidden');
  resultActions.classList.add('hidden');

  if (state === 'empty') emptyState.classList.remove('hidden');
  if (state === 'loading') loadingState.classList.remove('hidden');
  if (state === 'success') { resultsList.classList.remove('hidden'); resultActions.classList.remove('hidden'); }
  if (state === 'error') errorState.classList.remove('hidden');
};

const bytesToMB = (bytes) => (bytes / (1024 * 1024)).toFixed(1) + " MB";

function showToast(message, type = 'info'){
  toastMessage.textContent = message;
  if (type === 'success'){
    toastIcon.innerHTML = '<svg class="w-5 h-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>';
    toast.className = 'toast bg-green-900/90 border border-green-600 rounded-2xl px-6 py-4 shadow-lg';
  } else if (type === 'error'){
    toastIcon.innerHTML = '<svg class="w-5 h-5 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>';
    toast.className = 'toast bg-red-900/90 border border-red-600 rounded-2xl px-6 py-4 shadow-lg';
  } else {
    toastIcon.innerHTML = '<svg class="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>';
    toast.className = 'toast bg-blue-900/90 border border-blue-600 rounded-2xl px-6 py-4 shadow-lg';
  }
  // show
  requestAnimationFrame(()=> toast.classList.add('show'));
  // hide
  setTimeout(()=> toast.classList.remove('show'), 3000);
}

const setImagePreview = (dataURL, file) => {
  // Desktop
  previewImg.src = dataURL;
  previewImg.onload = () => imageDimensions.textContent = `${previewImg.naturalWidth}×${previewImg.naturalHeight}`;
  fileName.textContent = file.name;
  fileSize.textContent = bytesToMB(file.size);

  // Mobile
  previewImgMobile.src = dataURL;
  previewImgMobile.onload = () => imageDimensionsMobile.textContent = `${previewImgMobile.naturalWidth}×${previewImgMobile.naturalHeight}`;
  fileNameMobile.textContent = file.name;
  fileSizeMobile.textContent = bytesToMB(file.size);

  // Toggle views
  dropzone.classList.add('hidden');
  dropzoneMobile.classList.add('hidden');
  imagePreview.classList.remove('hidden');
  imagePreviewMobile.classList.remove('hidden');
  generateBtn.disabled = false;
};

const clearImage = () => {
  currentImage = null; generatedTitles = [];
  dropzone.classList.remove('hidden');
  dropzoneMobile.classList.remove('hidden');
  imagePreview.classList.add('hidden');
  imagePreviewMobile.classList.add('hidden');
  fileInput.value = ''; fileInputMobile.value = '';
  generateBtn.disabled = true;
  showResultsState('empty');
  statusRow.classList.add('hidden');
};

async function checkAPIConnection() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json'
      }
    });
    return response.ok;
  } catch (error) {
    console.error('API connection check failed:', error);
    return false;
  }
}

//  Handlers 
function handleFile(file){
  if (!file) return;
  if (!['image/jpeg','image/jpg','image/png'].includes(file.type)){
    showToast('Định dạng không được hỗ trợ. Hãy dùng JPG hoặc PNG.', 'error'); return;
  }
  if (file.size > 20*1024*1024){
    showToast('Ảnh quá lớn (>20MB). Vui lòng chọn ảnh nhỏ hơn.', 'error'); return;
  }
  currentImage = file;
  const reader = new FileReader();
  reader.onload = (e)=> setImagePreview(e.target.result, file);
  reader.readAsDataURL(file);
}

// Drag & drop (desktop)
['dragover','dragleave','drop'].forEach(evt=>{
  dropzone.addEventListener(evt, (e)=>{
    e.preventDefault();
    if(evt==='dragover') dropzone.classList.add('dropzone-active');
    if(evt==='dragleave') dropzone.classList.remove('dropzone-active');
    if(evt==='drop'){ dropzone.classList.remove('dropzone-active'); const f=e.dataTransfer.files; if(f.length) handleFile(f[0]); }
  });
});
// Drag & drop (mobile)
['dragover','dragleave','drop'].forEach(evt=>{
  dropzoneMobile.addEventListener(evt, (e)=>{
    e.preventDefault();
    if(evt==='dragover') dropzoneMobile.classList.add('dropzone-active');
    if(evt==='dragleave') dropzoneMobile.classList.remove('dropzone-active');
    if(evt==='drop'){ dropzoneMobile.classList.remove('dropzone-active'); const f=e.dataTransfer.files; if(f.length) handleFile(f[0]); }
  });
});

// Click upload
dropzone.addEventListener('click', ()=> fileInput.click());
uploadBtn.addEventListener('click', (e)=>{ e.stopPropagation(); fileInput.click(); });
fileInput.addEventListener('change', (e)=> e.target.files.length && handleFile(e.target.files[0]));

dropzoneMobile.addEventListener('click', ()=> fileInputMobile.click());
uploadBtnMobile.addEventListener('click', (e)=>{ e.stopPropagation(); fileInputMobile.click(); });
fileInputMobile.addEventListener('change', (e)=> e.target.files.length && handleFile(e.target.files[0]));

// Paste
document.addEventListener('paste', (e)=>{
  const items = e.clipboardData?.items || [];
  for (const it of items){
    if (it.type.indexOf('image') !== -1) { handleFile(it.getAsFile()); break; }
  }
});

// Change/remove
changeImageBtn.addEventListener('click', ()=> fileInput.click());
changeImageBtnMobile.addEventListener('click', ()=> fileInputMobile.click());
removeImageBtn.addEventListener('click', clearImage);
removeImageBtnMobile.addEventListener('click', clearImage);

// Slider
titleCount.addEventListener('input', (e)=> countBadge.textContent = e.target.value);

// Generate (call API /caption) 
generateBtn.addEventListener('click', async ()=>{
  if (!currentImage) return;

  const num = parseInt(titleCount.value, 10);
  const apiConnected = await checkAPIConnection();
  if (!apiConnected) {
    showToast('Không thể kết nối đến máy chủ. Vui lòng kiểm tra API server.', 'error');
    return;
  }

  const form = new FormData();
  form.append('file', currentImage);

  showResultsState('loading');
  statusRow.classList.remove('hidden');
  loadingBar.classList.remove('hidden');
  statusText.textContent = 'Đang xử lý…';

  const t0 = performance.now();
  try{
    const url = `${API_BASE_URL}/caption?n=${num}`;
    console.log(url);
    const res = await fetch(url, { 
      method:'POST', 
      body: form,
      headers: {
        'Accept': 'application/json'
      }
    });

    console.log('Response status:', res.status);
    console.log('Response headers:', Object.fromEntries(res.headers.entries()));
    
    if (!res.ok){
      let errorMessage = 'Lỗi không xác định';
      let errorDetail = '';
      
      try {
        const errorData = await res.json();
        errorMessage = errorData.detail || errorData.message || 'Caption generation failed';
        errorDetail = JSON.stringify(errorData);
      } catch (parseError) {
        try {
          errorMessage = await res.text() || `HTTP ${res.status}: ${res.statusText}`;
        } catch (textError) {
          errorMessage = `HTTP ${res.status}: ${res.statusText}`;
        }
      }
      
      console.error('API Error:', { status: res.status, message: errorMessage, detail: errorDetail });
      throw new Error(errorMessage);
    }
    
    const data = await res.json();
    console.log('API Response:', data);
    
    generatedTitles = Array.isArray(data.captions) ? data.captions : [];
    const latency = ((performance.now() - t0)/1000).toFixed(2);

    showResultsState('success');
    loadingBar.classList.add('hidden');
    statusText.textContent = `Thời gian xử lý: ${latency} giây · Số tiêu đề đã sinh: ${generatedTitles.length}`;
    renderResults();
    
  } catch(err){
    console.error('Full error object:', err);
    console.error('Error stack:', err.stack);
    
    showResultsState('error');
    loadingBar.classList.add('hidden');
    statusText.textContent = '';
    
    let userMessage = 'Không thể xử lý ảnh. Vui lòng thử lại.';
    
    if (err.name === 'TypeError' && err.message.includes('fetch')) {
      userMessage = 'Lỗi kết nối. Kiểm tra API server có đang chạy không.';
    } else if (err.message.includes('CORS')) {
      userMessage = 'Lỗi CORS. Kiểm tra cấu hình server.';
    } else if (err.message.includes('Network')) {
      userMessage = 'Lỗi mạng. Kiểm tra kết nối internet.';
    } else if (err.message && err.message !== 'Caption generation failed') {
      userMessage = err.message;
    }
    
    showToast(userMessage, 'error');
  }
});

// Clear
clearBtn.addEventListener('click', clearImage);

// Results render
function renderResults(){
  resultsList.innerHTML = '';
  generatedTitles.forEach((title, idx)=>{
    const wrap = document.createElement('div');
    wrap.className = 'flex items-start justify-between p-4 bg-gray-900/50 rounded-xl border border-gray-700';
    wrap.innerHTML = `
      <div class="flex-1 pr-4">
        <span class="inline-block w-6 h-6 bg-blue-500/20 text-blue-300 text-xs font-bold rounded-full text-center leading-6 mr-3">${idx+1}</span>
        <span class="text-gray-200">${title}</span>
      </div>
      <button data-idx="${idx}" class="copy-one flex items-center gap-2 text-sm bg-gray-700 hover:bg-gray-600 text-gray-300 px-3 py-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-gray-500">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"></path>
        </svg>
        <span>Sao chép</span>
      </button>
    `;
    resultsList.appendChild(wrap);
  });

  // Copy handlers
  resultsList.querySelectorAll('.copy-one').forEach(btn=>{
    btn.addEventListener('click', async (e)=>{
      const idx = parseInt(e.currentTarget.getAttribute('data-idx'),10);
      const title = generatedTitles[idx] || '';
      try{
        await navigator.clipboard.writeText(title);
        showToast('Đã sao chép vào clipboard.', 'success');
      }catch(_){}
    });
  });

  // Actions
  copyAllBtn.onclick = async ()=>{
    try{
      await navigator.clipboard.writeText(generatedTitles.join('\n'));
      showToast('Đã sao chép tất cả vào clipboard.', 'success');
    }catch(_){}
  };
  downloadBtn.onclick = ()=>{
    const data = { titles: generatedTitles, count: generatedTitles.length, timestamp: new Date().toISOString(),
      image_info: currentImage ? { name: currentImage.name, size: currentImage.size, type: currentImage.type } : null };
    const blob = new Blob([JSON.stringify(data,null,2)], {type:'application/json'});
    const url = URL.createObjectURL(blob); const a = document.createElement('a');
    a.href = url; a.download = `sports-titles-${Date.now()}.json`; document.body.appendChild(a); a.click();
    document.body.removeChild(a); URL.revokeObjectURL(url);
  };
}

// Keyboard access
document.addEventListener('keydown', (e)=>{ if(e.key==='Enter' && e.target===dropzone) fileInput.click(); });

// Init
countBadge.textContent = titleCount.value;
showResultsState('empty');

window.addEventListener('load', async () => {
  const connected = await checkAPIConnection();
  if (!connected) {
    showToast('Cảnh báo: Không thể kết nối đến API server', 'error');
  } else {
    console.log('API server connected successfully');
  }
});