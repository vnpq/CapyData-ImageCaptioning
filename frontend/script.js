const API_BASE = "http://0.0.0.0:8000"; 

let selectedModel = "cnn_lstm";
let captionCount  = 1;
let selectedFile  = null;

const dropzone   = document.getElementById("dropzone");
const fileInput  = document.getElementById("fileInput");
const preview    = document.getElementById("preview");
const resultsBox = document.getElementById("resultsBox");
const latencyTxt = document.getElementById("latencyText");
const imgSizeTxt = document.getElementById("imgSizeText");
const btnGen     = document.getElementById("generateBtn");

function loadFile(file){
  if(!file){ return; }
  if(!file.type.startsWith("image/")){
    dropzone.setAttribute("data-error", "Chỉ chấp nhận ảnh (JPEG/PNG).");
    return;
  }
  if(file.size > 10 * 1024 * 1024){
    dropzone.setAttribute("data-error", "Kích thước ảnh vượt quá 10MB.");
    return;
  }
  dropzone.removeAttribute("data-error");
  selectedFile = file;

  const reader = new FileReader();
  reader.onload = (e) => {
    preview.innerHTML = `<img src="${e.target.result}" alt="Uploaded image" class="preview-image"/>`;
    const img = new Image();
    img.onload = () => imgSizeTxt.textContent = `Image size: ${img.width} × ${img.height}`;
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

dropzone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (e) => loadFile(e.target.files[0]));

["dragenter","dragover"].forEach(evt => dropzone.addEventListener(evt, e => {
  e.preventDefault(); e.stopPropagation();
  dropzone.classList.add("dragover");
}));
["dragleave","drop"].forEach(evt => dropzone.addEventListener(evt, e => {
  e.preventDefault(); e.stopPropagation();
  dropzone.classList.remove("dragover");
}));
dropzone.addEventListener("drop", e => {
  const file = e.dataTransfer.files && e.dataTransfer.files[0];
  loadFile(file);
});

window.addEventListener("paste", e => {
  const items = e.clipboardData && e.clipboardData.items;
  if(!items) return;
  for(const it of items){
    if(it.type.indexOf("image") !== -1){
      const file = it.getAsFile();
      loadFile(file);
      break;
    }
  }
});

document.getElementById("modelSegment").addEventListener("click", (e) => {
  const btn = e.target.closest(".segment-btn");
  if(!btn) return;
  document.querySelectorAll(".segment-btn").forEach(b => b.classList.remove("is-active"));
  btn.classList.add("is-active");
  selectedModel = btn.dataset.model;
});

document.getElementById("countPills").addEventListener("click", (e) => {
  const pill = e.target.closest(".pill");
  if(!pill) return;
  document.querySelectorAll(".pill").forEach(p => p.classList.remove("is-active"));
  pill.classList.add("is-active");
  captionCount = Number(pill.dataset.count || 1);
});

document.getElementById("btnRefresh").addEventListener("click", () => {
  btnToast("Đã yêu cầu refresh models.");
});

async function generateCaptions(){
  if(!selectedFile){
    btnToast("Vui lòng chọn ảnh trước.");
    return;
  }

  btnGen.disabled = true;
  const prevLabel = btnGen.textContent;
  btnGen.innerHTML = `<div class="spinner" style="width:16px;height:16px;border:2px solid transparent;border-top:2px solid currentColor;border-radius:50%;animation:spin 1s linear infinite"></div> Generating...`;
  resultsBox.innerHTML = Array.from({length:captionCount||1}, () => `<div class="ghost-line"></div>`).join("\n");

  const url = `${API_BASE}/caption?model=${encodeURIComponent(selectedModel)}&num_captions=${captionCount}`;

  const fd = new FormData();
  fd.append("file", selectedFile, selectedFile.name);

  try{
    const res = await fetch(url, { method: "POST", body: fd });
    if(!res.ok){
      const text = await res.text();
      throw new Error(text || res.statusText);
    }
    const data = await res.json();
    const caps = Array.isArray(data.captions) ? data.captions : [];
    resultsBox.textContent = caps.length ? caps.map((c,i)=>`${i+1}. ${c}`).join("\n") : "(Không có caption trả về)";
    latencyTxt.textContent = `⏱ Latency: ${Number(data.latency_ms||0).toFixed(1)} ms`;
  }catch(err){
    resultsBox.textContent = `Lỗi API: ${String(err.message||err)}`;
    latencyTxt.textContent = "⏱ Latency: —";
  }finally{
    btnGen.disabled = false;
    btnGen.textContent = prevLabel;
  }
}
btnGen.addEventListener("click", generateCaptions);

document.getElementById("btnCopy").addEventListener("click", async () => {
  try{
    await navigator.clipboard.writeText(resultsBox.textContent || "");
    btnToast("Đã sao chép");
  }catch{ btnToast("Không thể sao chép"); }
});

document.getElementById("btnDownload").addEventListener("click", () => {
  const blob = new Blob([resultsBox.textContent || ""], {type:"text/plain;charset=utf-8"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "captions.txt";
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(()=>URL.revokeObjectURL(a.href), 1000);
});

function btnToast(msg){
  const t = document.createElement("div");
  t.textContent = msg;
  t.style.position="fixed";
  t.style.bottom="20px";
  t.style.right="20px";
  t.style.background="#263043";
  t.style.color="#E5E7EB";
  t.style.border="1px solid #3a465e";
  t.style.padding="10px 12px";
  t.style.borderRadius="8px";
  t.style.fontSize="13px";
  t.style.zIndex="9999";
  document.body.appendChild(t);
  setTimeout(()=>t.remove(), 1400);
}

const style = document.createElement("style");
style.innerHTML = `@keyframes spin{to{transform:rotate(360deg)}}`;
document.head.appendChild(style);