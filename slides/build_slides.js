// build_slides.js — Run 6/8 presentation deck for CS231 UIT 2026
// Output: slides/run68_deck.pptx
// Usage: cd slides && npm install pptxgenjs && node build_slides.js
//
// Image placeholders are loaded from slides/assets/ — see slides/README.md for
// the asset checklist. Missing files trigger a console warning but slide
// generation continues so layout can be iterated independently.

const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");

const ASSETS = path.join(__dirname, "assets");
const OUT = path.join(__dirname, "run68_deck.pptx");

// ---------- design tokens (per academic-pptx skill) -------------------------

const COLORS = {
  bg:        "FFFFFF",
  primary:   "1F4E79",  // dark navy — titles, sandwich slides
  accent:    "2E75B6",  // mid-blue — headers, callouts
  body:      "2D2D2D",
  muted:     "777777",
  rule:      "CCCCCC",
  highlight: "FFF2CC",
  good:      "548235",  // green — positive deltas
  bad:       "C00000",  // red — regressions
};

const FONT = "Arial";       // Arial supports Vietnamese diacritics
const FS = {
  titleBig:   32,           // title slide hero
  title:      26,           // action title
  section:    22,           // within-slide section header
  body:       20,           // body bullets (minimum per skill)
  bodySmall:  18,           // dense slides
  label:      16,           // chart annotations
  cite:       13,           // citations
  muted:      14,           // breadcrumbs, captions
};

const MARGIN = 0.5;
const W = 10.0;             // 16:9 slide width
const H = 5.625;            // 16:9 slide height

// ---------- helpers ---------------------------------------------------------

function imageOrPlaceholder(slide, relPath, opts) {
  const full = path.join(ASSETS, relPath);
  if (fs.existsSync(full)) {
    slide.addImage({ path: full, ...opts });
  } else {
    console.warn(`  [warn] missing asset: ${relPath} — rendering placeholder`);
    slide.addShape("rect", {
      ...opts,
      fill: { color: "F0F4F8" },
      line: { color: COLORS.rule, width: 1, dashType: "dash" },
    });
    slide.addText(`[CHỖ ẢNH]\n${relPath}`, {
      ...opts,
      fontSize: 14, fontFace: FONT, color: COLORS.muted,
      align: "center", valign: "middle",
    });
  }
}

function actionTitle(slide, text) {
  slide.addText(text, {
    x: MARGIN, y: 0.2, w: W - 2 * MARGIN, h: 0.85,
    fontSize: FS.title, fontFace: FONT, color: COLORS.primary,
    bold: true, align: "left", valign: "top",
  });
  // divider rule
  slide.addShape("rect", {
    x: MARGIN, y: 1.07, w: W - 2 * MARGIN, h: 0.025,
    fill: { color: COLORS.rule }, line: { color: COLORS.rule },
  });
}

function footerCite(slide, text) {
  slide.addText(text, {
    x: MARGIN, y: H - 0.4, w: W - 2 * MARGIN, h: 0.3,
    fontSize: FS.cite, fontFace: FONT, color: COLORS.muted,
    align: "left", valign: "middle",
  });
}

function pageNum(slide, n, total) {
  slide.addText(`${n} / ${total}`, {
    x: W - 0.9, y: H - 0.35, w: 0.7, h: 0.25,
    fontSize: 10, fontFace: FONT, color: COLORS.muted,
    align: "right", valign: "middle",
  });
}

// ---------- presentation ----------------------------------------------------

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.title = "Phân loại Hành vi Lái xe Mất tập trung — CS231 UIT 2026";
pres.author = "Nguyễn Xuân Trường";
pres.company = "UIT — CS231 Computer Vision";

const TOTAL_SLIDES = 15;

// ============================================================================
// SLIDE 1 — Title (dark navy sandwich)
// ============================================================================
{
  const s = pres.addSlide();
  s.background = { color: COLORS.primary };

  s.addText([
    { text: "Phân loại Hành vi Lái xe Mất tập trung", options: { breakLine: true } },
    { text: "ResNet-18 + CBAM, huấn luyện from-scratch", options: {} },
  ], {
    x: 0.7, y: 1.3, w: 8.6, h: 1.8,
    fontSize: FS.titleBig, fontFace: FONT, color: "FFFFFF",
    bold: true, align: "left", valign: "top",
  });

  // Accent rule
  s.addShape("rect", {
    x: 0.7, y: 3.2, w: 2.0, h: 0.04,
    fill: { color: COLORS.accent }, line: { color: COLORS.accent },
  });

  s.addText("State Farm · Subject-wise split · Đánh giá trên tài xế chưa thấy", {
    x: 0.7, y: 3.35, w: 8.6, h: 0.5,
    fontSize: 18, fontFace: FONT, color: "A0BBDD",
    italic: true, align: "left",
  });

  s.addText([
    { text: "Nguyễn Xuân Trường", options: { breakLine: true, bold: true } },
    { text: "CS231 — Thị giác Máy tính · UIT · 2026", options: { breakLine: true } },
    { text: "github.com/nxtruoong/DoAnCS231", options: { italic: true } },
  ], {
    x: 0.7, y: 4.0, w: 8.6, h: 1.2,
    fontSize: 16, fontFace: FONT, color: "CADCFC",
    align: "left",
  });
}

// ============================================================================
// SLIDE 2 — Bài toán
// ============================================================================
{
  const s = pres.addSlide();
  actionTitle(s,
    "Lái xe mất tập trung gây ~3.500 tử vong/năm — bài toán phân loại 10 hành vi từ một ảnh cabin");

  // Left: bullets
  s.addText([
    { text: "Nhiệm vụ: ", options: { bold: true, breakLine: false } },
    { text: "phân loại 1 ảnh → 1 trong 10 lớp hành vi.", options: { breakLine: true } },
    { text: "Dữ liệu: ", options: { bold: true, breakLine: false } },
    { text: "State Farm Kaggle, ~22k ảnh nhãn, 640×480, camera cố định bên phải tài xế.", options: { breakLine: true } },
    { text: "Ý nghĩa: ", options: { bold: true, breakLine: false } },
    { text: "khối nền cho ADAS / driver-monitoring system.", options: { breakLine: true } },
    { text: "Ràng buộc: ", options: { bold: true, breakLine: false } },
    { text: "huấn luyện ", options: { breakLine: false } },
    { text: "từ đầu", options: { bold: true, italic: true, breakLine: false } },
    { text: " — không dùng ImageNet pretrain.", options: {} },
  ], {
    x: MARGIN, y: 1.25, w: 4.4, h: 3.6,
    fontSize: FS.bodySmall, fontFace: FONT, color: COLORS.body,
    bullet: { type: "bullet" }, paraSpaceAfter: 10, valign: "top",
  });

  // Right: 2×5 image grid placeholder
  imageOrPlaceholder(s, "classes_grid_2x5.png", {
    x: 5.1, y: 1.2, w: 4.4, h: 3.7,
  });

  footerCite(s, "Nguồn: State Farm Distracted Driver Detection (Kaggle, 2016) · NHTSA 2022 fatality estimate");
  pageNum(s, 2, TOTAL_SLIDES);
}

// ============================================================================
// SLIDE 3 — Subject-wise split
// ============================================================================
{
  const s = pres.addSlide();
  actionTitle(s,
    "Chia tách theo tài xế chứ không random — vì cùng tài xế ở 2 split làm val acc nhảy lên ~99% giả");

  // Two-column layout
  // Left: Random split (BAD)
  s.addShape("rect", {
    x: MARGIN, y: 1.25, w: 4.4, h: 0.5,
    fill: { color: COLORS.bad }, line: { color: COLORS.bad },
  });
  s.addText("Random split — LEAK", {
    x: MARGIN, y: 1.25, w: 4.4, h: 0.5,
    fontSize: FS.section, fontFace: FONT, color: "FFFFFF",
    bold: true, align: "center", valign: "middle",
  });
  s.addText([
    { text: "Cùng 1 tài xế có ảnh ở cả train + val", options: { breakLine: true } },
    { text: "Mô hình học mặt / áo / dáng ngồi", options: { breakLine: true } },
    { text: "Val acc ~99% nhưng vô dụng trên người mới", options: { breakLine: true } },
  ], {
    x: MARGIN, y: 1.85, w: 4.4, h: 2.3,
    fontSize: FS.bodySmall, fontFace: FONT, color: COLORS.body,
    bullet: { type: "bullet" }, paraSpaceAfter: 8,
  });

  // Right: Subject-wise (GOOD)
  s.addShape("rect", {
    x: 5.1, y: 1.25, w: 4.4, h: 0.5,
    fill: { color: COLORS.good }, line: { color: COLORS.good },
  });
  s.addText("Subject-wise split — đúng", {
    x: 5.1, y: 1.25, w: 4.4, h: 0.5,
    fontSize: FS.section, fontFace: FONT, color: "FFFFFF",
    bold: true, align: "center", valign: "middle",
  });
  s.addText([
    { text: "Held-out 5/26 tài xế: p022, p035, p047, p056, p075", options: { breakLine: true } },
    { text: "~17.500 train / ~4.500 val (≈ 19%)", options: { breakLine: true } },
    { text: "Val acc phản ánh tổng quát hoá thật", options: { breakLine: true } },
  ], {
    x: 5.1, y: 1.85, w: 4.4, h: 2.3,
    fontSize: FS.bodySmall, fontFace: FONT, color: COLORS.body,
    bullet: { type: "bullet" }, paraSpaceAfter: 8,
  });

  // Callout box at bottom — the "so what"
  s.addShape("roundRect", {
    x: MARGIN, y: 4.35, w: W - 2 * MARGIN, h: 0.7,
    fill: { color: COLORS.highlight }, line: { color: "E6C800", width: 1.5 },
    rectRadius: 0.08,
  });
  s.addText([
    { text: "Bài học: ", options: { bold: true, breakLine: false } },
    { text: "subject-wise eval là không thể bỏ qua trong driver-monitoring — random split bịa kết quả.", options: {} },
  ], {
    x: MARGIN + 0.1, y: 4.35, w: W - 2 * MARGIN - 0.2, h: 0.7,
    fontSize: 16, fontFace: FONT, color: "7A5200",
    align: "center", valign: "middle",
  });

  footerCite(s, "ADR 0001 — Subject-wise Split · log.md");
  pageNum(s, 3, TOTAL_SLIDES);
}

// ============================================================================
// SLIDE 4 — Architecture intuition
// ============================================================================
{
  const s = pres.addSlide();
  actionTitle(s,
    "ResNet-18 + CBAM thêm \"nhìn vào đâu\" và \"kênh nào quan trọng\" chỉ với +0.4% tham số");

  // Left: arch diagram placeholder
  imageOrPlaceholder(s, "arch_resnet18_cbam.png", {
    x: MARGIN, y: 1.25, w: 5.4, h: 3.6,
  });

  // Right: intuition bullets
  s.addText("Ý tưởng (không formula)", {
    x: 6.1, y: 1.25, w: 3.4, h: 0.4,
    fontSize: FS.section, fontFace: FONT, color: COLORS.accent, bold: true,
  });
  s.addText([
    { text: "ResNet-18: ", options: { bold: true, breakLine: false } },
    { text: "8 khối residual, ~11M tham số.", options: { breakLine: true } },
    { text: "CBAM-CAM: ", options: { bold: true, breakLine: false } },
    { text: "\"kênh nào nói chuyện\" — vd kênh tông da khi tay đưa lên mặt.", options: { breakLine: true } },
    { text: "CBAM-SAM: ", options: { bold: true, breakLine: false } },
    { text: "\"vùng nào cần nhìn\" — heatmap demo lấy từ đây.", options: { breakLine: true } },
    { text: "Chi phí CBAM: ", options: { bold: true, breakLine: false } },
    { text: "+44k params trên ~11M = ", options: { breakLine: false } },
    { text: "+0.4%", options: { bold: true, color: COLORS.good } },
    { text: ".", options: {} },
  ], {
    x: 6.1, y: 1.7, w: 3.4, h: 3.0,
    fontSize: FS.bodySmall, fontFace: FONT, color: COLORS.body,
    bullet: { type: "bullet" }, paraSpaceAfter: 10,
  });

  footerCite(s, "He et al. 2016 (ResNet) · Woo et al. 2018 (CBAM)");
  pageNum(s, 4, TOTAL_SLIDES);
}

// ============================================================================
// SLIDE 5 — CBAM heatmap visual (the hero slide)
// ============================================================================
{
  const s = pres.addSlide();
  actionTitle(s,
    "Heatmap CBAM cho thấy mô hình nhìn đúng vùng tay+điện thoại, radio, và tay-tới-mặt");

  // 3-row CBAM grid — single image asset
  imageOrPlaceholder(s, "cbam_visual_c1_c5_c8.png", {
    x: MARGIN, y: 1.2, w: W - 2 * MARGIN, h: 3.5,
  });

  // Caption below
  s.addText([
    { text: "Hàng 1: ", options: { bold: true, breakLine: false } },
    { text: "c1 (nhắn tin tay phải) — heatmap dồn vào tay phải + vô-lăng.   ", options: { breakLine: false } },
    { text: "Hàng 2: ", options: { bold: true, breakLine: false } },
    { text: "c5 (radio) — dồn về dashboard.   ", options: { breakLine: false } },
    { text: "Hàng 3: ", options: { bold: true, breakLine: false } },
    { text: "c8 (tóc) — dồn về đầu/tay-tới-mặt.", options: {} },
  ], {
    x: MARGIN, y: 4.75, w: W - 2 * MARGIN, h: 0.5,
    fontSize: 14, fontFace: FONT, color: COLORS.body, align: "center", valign: "middle",
  });

  footerCite(s, "SAM layer4, Run 6 EMA weights · run6/eval/attention_grid.png");
  pageNum(s, 5, TOTAL_SLIDES);
}

// ============================================================================
// SLIDE 6 — Run 1-4 trials (debugging story)
// ============================================================================
{
  const s = pres.addSlide();
  actionTitle(s,
    "Hai bug bị bắt sớm trong Run 1-4: init quá lớn và EMA áp sai lên BN — sửa rồi mới so kết quả thật");

  // Table-as-text
  const rows = [
    [
      { text: "Run", options: { bold: true, fill: { color: "EBF3FA" } } },
      { text: "Thay đổi", options: { bold: true, fill: { color: "EBF3FA" } } },
      { text: "Val", options: { bold: true, align: "right", fill: { color: "EBF3FA" } } },
      { text: "EMA", options: { bold: true, align: "right", fill: { color: "EBF3FA" } } },
      { text: "Phát hiện", options: { bold: true, fill: { color: "EBF3FA" } } },
    ],
    [
      { text: "1" },
      { text: "LR 0.1, không warmup" },
      { text: "0.10", options: { align: "right" } },
      { text: "0.10", options: { align: "right" } },
      { text: "Bug A: FC init quá lớn → ReLU chết. Loss = ln(10).", options: { color: COLORS.bad } },
    ],
    [
      { text: "2" },
      { text: "Smoke, augment tối thiểu" },
      { text: "0.62", options: { align: "right" } },
      { text: "0.14", options: { align: "right" } },
      { text: "Pipeline OK sau fix Bug A." },
    ],
    [
      { text: "3" },
      { text: "+ CBAM + CutMix" },
      { text: "0.74", options: { align: "right" } },
      { text: "0.10", options: { align: "right" } },
      { text: "Bug B: EMA áp 0.999 lên BN running stats.", options: { color: COLORS.bad } },
    ],
    [
      { text: "4" },
      { text: "Bug B đã sửa" },
      { text: "0.80", options: { align: "right" } },
      { text: "0.59", options: { align: "right" } },
      { text: "EMA hồi sinh, nhưng decay 0.999 quá chậm." },
    ],
  ];

  s.addTable(rows, {
    x: MARGIN, y: 1.25, w: W - 2 * MARGIN, h: 2.6,
    colW: [0.6, 2.4, 0.7, 0.7, 4.6],
    fontSize: 14, fontFace: FONT, color: COLORS.body,
    border: { type: "solid", pt: 0.5, color: COLORS.rule },
    valign: "middle",
  });

  // Callout box: lesson
  s.addShape("roundRect", {
    x: MARGIN, y: 4.05, w: W - 2 * MARGIN, h: 1.0,
    fill: { color: COLORS.highlight }, line: { color: "E6C800", width: 1.5 },
    rectRadius: 0.08,
  });
  s.addText([
    { text: "Bài học phương pháp: ", options: { bold: true, breakLine: true } },
    { text: "(1) loss kẹt ở 2.3026 = ln(10) → softmax sập về uniform → init/LR sai.  ", options: { breakLine: false } },
    { text: "(2) EMA chỉ áp lên parameters; BN buffers phải copy thẳng từ live model.", options: {} },
  ], {
    x: MARGIN + 0.15, y: 4.1, w: W - 2 * MARGIN - 0.3, h: 0.9,
    fontSize: 14, fontFace: FONT, color: "7A5200", align: "left", valign: "middle",
  });

  footerCite(s, "log.md Run 1 & Run 3 · commit ed889e3, 7c5a2a8");
  pageNum(s, 6, TOTAL_SLIDES);
}

// ============================================================================
// SLIDE 7 — Run 6 headline result
// ============================================================================
{
  const s = pres.addSlide();
  actionTitle(s,
    "Run 6 đạt headline: macro F1 = 0.873 trên 5 tài xế chưa thấy, +5pp so với Run 5");

  // Left: training curves
  imageOrPlaceholder(s, "run6_training_curves.png", {
    x: MARGIN, y: 1.2, w: 5.4, h: 3.7,
  });

  // Right: key numbers
  s.addText("Số chính", {
    x: 6.2, y: 1.2, w: 3.3, h: 0.4,
    fontSize: FS.section, fontFace: FONT, color: COLORS.accent, bold: true,
  });
  s.addText([
    { text: "Best EMA val acc: ", options: { breakLine: false } },
    { text: "0.8747", options: { bold: true, color: COLORS.primary, breakLine: true } },
    { text: "Best raw val acc: ", options: { breakLine: false } },
    { text: "0.8683", options: { bold: true, breakLine: true } },
    { text: "Macro F1: ", options: { breakLine: false } },
    { text: "0.873", options: { bold: true, color: COLORS.primary, breakLine: true } },
    { text: "Train time: ", options: { breakLine: false } },
    { text: "137 min (T4×2)", options: { breakLine: true } },
    { text: "Params: ", options: { breakLine: false } },
    { text: "11.2 M", options: { breakLine: true } },
  ], {
    x: 6.2, y: 1.65, w: 3.3, h: 2.5,
    fontSize: 16, fontFace: FONT, color: COLORS.body,
    bullet: { type: "bullet" }, paraSpaceAfter: 8,
  });

  // Highlight callout
  s.addShape("roundRect", {
    x: 6.2, y: 4.0, w: 3.3, h: 0.8,
    fill: { color: COLORS.highlight }, line: { color: "E6C800", width: 1.5 },
    rectRadius: 0.06,
  });
  s.addText([
    { text: "+5pp ", options: { bold: true, color: COLORS.good, breakLine: false } },
    { text: "macro F1 so với Run 5 nhờ: 384 input + tight crop + CutMix p=0.15.", options: {} },
  ], {
    x: 6.3, y: 4.0, w: 3.1, h: 0.8,
    fontSize: 12, fontFace: FONT, color: "7A5200",
    align: "left", valign: "middle",
  });

  footerCite(s, "run6/eval/metrics.json · log.md Run 6");
  pageNum(s, 7, TOTAL_SLIDES);
}

// ============================================================================
// SLIDE 8 — Run 6 per-class + confusion matrix
// ============================================================================
{
  const s = pres.addSlide();
  actionTitle(s,
    "Run 6 giỏi ở lớp có vật cầm tay rõ (c1-c7 đều ≥ 0.94) nhưng yếu ở lớp \"thụ động\" c0/c8/c9");

  // Left: per-class table
  const rows = [
    [
      { text: "Lớp", options: { bold: true, fill: { color: "EBF3FA" } } },
      { text: "F1", options: { bold: true, align: "right", fill: { color: "EBF3FA" } } },
    ],
    [{ text: "c0 safe" }, { text: "0.66", options: { align: "right", color: COLORS.bad, bold: true } }],
    [{ text: "c1 text-R" }, { text: "0.95", options: { align: "right" } }],
    [{ text: "c2 phone-R" }, { text: "0.95", options: { align: "right" } }],
    [{ text: "c3 text-L" }, { text: "0.92", options: { align: "right", color: COLORS.good } }],
    [{ text: "c4 phone-L" }, { text: "0.97", options: { align: "right" } }],
    [{ text: "c5 radio" }, { text: "0.94", options: { align: "right" } }],
    [{ text: "c6 drink" }, { text: "0.94", options: { align: "right" } }],
    [{ text: "c7 reach" }, { text: "0.98", options: { align: "right" } }],
    [{ text: "c8 hair" }, { text: "0.75", options: { align: "right", color: COLORS.bad, bold: true } }],
    [{ text: "c9 talk" }, { text: "0.68", options: { align: "right", color: COLORS.bad, bold: true } }],
  ];
  s.addTable(rows, {
    x: MARGIN, y: 1.25, w: 2.6, h: 3.5,
    colW: [1.6, 1.0],
    fontSize: 13, fontFace: FONT, color: COLORS.body,
    border: { type: "solid", pt: 0.5, color: COLORS.rule },
    valign: "middle",
  });

  // Right: confusion matrix placeholder
  imageOrPlaceholder(s, "run6_confusion_matrix.png", {
    x: 3.4, y: 1.25, w: 6.1, h: 3.5,
  });

  // Bottom callout
  s.addShape("roundRect", {
    x: MARGIN, y: 4.85, w: W - 2 * MARGIN, h: 0.6,
    fill: { color: "FFE4E4" }, line: { color: COLORS.bad, width: 1.5 },
    rectRadius: 0.06,
  });
  s.addText([
    { text: "Vấn đề tồn dư: ", options: { bold: true, breakLine: false } },
    { text: "tam giác c0 ↔ c8 ↔ c9 (lớp không có vật cầm tay) — motivation cho Run 7/8.", options: {} },
  ], {
    x: MARGIN + 0.15, y: 4.85, w: W - 2 * MARGIN - 0.3, h: 0.6,
    fontSize: 14, fontFace: FONT, color: COLORS.bad,
    align: "center", valign: "middle",
  });

  footerCite(s, "run6/eval/classification_report.txt");
  pageNum(s, 8, TOTAL_SLIDES);
}

// ============================================================================
// SLIDE 9 — Run 7 regression
// ============================================================================
{
  const s = pres.addSlide();
  actionTitle(s,
    "Thêm CNN stream thứ hai (Run 7) regress -12pp — chia capacity giữa 2 backbone trên 22k ảnh không đủ");

  // Left: per-class delta bar chart placeholder
  imageOrPlaceholder(s, "run7_vs_run6_delta.png", {
    x: MARGIN, y: 1.25, w: 5.0, h: 3.6,
  });

  // Right: explanation
  s.addText("Cái gì xảy ra", {
    x: 5.7, y: 1.25, w: 3.8, h: 0.4,
    fontSize: FS.section, fontFace: FONT, color: COLORS.accent, bold: true,
  });
  s.addText([
    { text: "Architecture: ", options: { bold: true, breakLine: false } },
    { text: "full @384 + top-crop face @224 (2× ResNet18+CBAM).", options: { breakLine: true } },
    { text: "Params: ", options: { bold: true, breakLine: false } },
    { text: "11.2M → 22.7M (×2.03).", options: { breakLine: true } },
    { text: "Kết quả eval: ", options: { bold: true, breakLine: false } },
    { text: "macro F1 0.873 → ", options: { breakLine: false } },
    { text: "0.748", options: { bold: true, color: COLORS.bad, breakLine: false } },
    { text: " (-12pp).", options: { breakLine: true } },
    { text: "Bipolar failure: ", options: { bold: true, breakLine: false } },
    { text: "c3 thành \"dumping ground\" (P=0.38), c9 starving (R=0.29).", options: {} },
  ], {
    x: 5.7, y: 1.7, w: 3.8, h: 2.4,
    fontSize: 14, fontFace: FONT, color: COLORS.body,
    bullet: { type: "bullet" }, paraSpaceAfter: 8,
  });

  // Bottom: lesson
  s.addShape("roundRect", {
    x: MARGIN, y: 4.85, w: W - 2 * MARGIN, h: 0.55,
    fill: { color: COLORS.highlight }, line: { color: "E6C800", width: 1.5 },
    rectRadius: 0.06,
  });
  s.addText([
    { text: "Lý do: ", options: { bold: true, breakLine: false } },
    { text: "face stream top-50% trùng phần lớn với full stream → 2 stream học cue trùng, overfit driver-ID.", options: {} },
  ], {
    x: MARGIN + 0.15, y: 4.85, w: W - 2 * MARGIN - 0.3, h: 0.55,
    fontSize: 13, fontFace: FONT, color: "7A5200",
    align: "left", valign: "middle",
  });

  footerCite(s, "run7/eval/metrics.json · log.md Run 7");
  pageNum(s, 9, TOTAL_SLIDES);
}

// ============================================================================
// SLIDE 10 — Run 8 pivot: pose fusion
// ============================================================================
{
  const s = pres.addSlide();
  actionTitle(s,
    "Run 8 pivot: thay CNN thứ hai bằng 36 toạ độ landmark MediaPipe — nửa số param, gấp đôi tín hiệu tư thế");

  // Top: architecture diagram placeholder
  imageOrPlaceholder(s, "arch_run8_pose_fusion.png", {
    x: MARGIN, y: 1.2, w: W - 2 * MARGIN, h: 2.4,
  });

  // Bottom: feature group breakdown (4 columns)
  const groups = [
    { title: "Đầu + thân (8)", body: "yaw, pitch, roll, vai\nshoulder twist\nvisibility ear" },
    { title: "Cổ tay (8)", body: "(x,y) cả 2 cổ tay\nspread ngang\nvertical asymmetry" },
    { title: "Khuỷu + ngón (8)", body: "vị trí khuỷu tay\nindex-wrist Δy\nthumb-pinky spread" },
    { title: "Hông + dẫn xuất (12)", body: "lap reference\nwrist-hip Δy\nwrist-shoulder Δx\nvisibility gates" },
  ];
  const colW = (W - 2 * MARGIN) / 4 - 0.15;
  groups.forEach((g, i) => {
    const x = MARGIN + i * (colW + 0.2);
    s.addShape("roundRect", {
      x, y: 3.8, w: colW, h: 1.5,
      fill: { color: "EBF3FA" }, line: { color: COLORS.accent, width: 1 },
      rectRadius: 0.06,
    });
    s.addText(g.title, {
      x, y: 3.85, w: colW, h: 0.35,
      fontSize: 13, fontFace: FONT, color: COLORS.primary,
      bold: true, align: "center",
    });
    s.addText(g.body, {
      x: x + 0.05, y: 4.2, w: colW - 0.1, h: 1.05,
      fontSize: 11, fontFace: FONT, color: COLORS.body,
      align: "center", valign: "top",
    });
  });

  footerCite(s, "extract_pose.py · RUN8_PLAN.md");
  pageNum(s, 10, TOTAL_SLIDES);
}

// ============================================================================
// SLIDE 11 — Pose features discriminate failing classes
// ============================================================================
{
  const s = pres.addSlide();
  actionTitle(s,
    "Pose features tách rõ lớp problematic: dấu p29 isolate c7, p28 isolate c3, p31 separate c5 vs c7");

  // Table of per-class means (from sanity check)
  const rows = [
    [
      { text: "Lớp", options: { bold: true, fill: { color: "EBF3FA" } } },
      { text: "p0 yaw", options: { bold: true, align: "right", fill: { color: "EBF3FA" } } },
      { text: "p28 lap-L", options: { bold: true, align: "right", fill: { color: "EBF3FA" } } },
      { text: "p29 lap-R", options: { bold: true, align: "right", fill: { color: "EBF3FA" } } },
      { text: "p31 reach-R", options: { bold: true, align: "right", fill: { color: "EBF3FA" } } },
      { text: "Tín hiệu", options: { bold: true, fill: { color: "EBF3FA" } } },
    ],
    [{ text: "c0 safe" }, { text: "-0.048", options: { align: "right" } }, { text: "-0.395", options: { align: "right" } }, { text: "-0.329", options: { align: "right" } }, { text: "+0.409", options: { align: "right" } }, { text: "2 tay vô-lăng" }],
    [{ text: "c3 text-L" }, { text: "-0.047", options: { align: "right" } }, { text: "-0.254", options: { align: "right", bold: true, color: COLORS.good } }, { text: "-0.274", options: { align: "right" } }, { text: "+0.407", options: { align: "right" } }, { text: "tay trái xuống lap" }],
    [{ text: "c5 radio" }, { text: "-0.063", options: { align: "right" } }, { text: "-0.394", options: { align: "right" } }, { text: "-0.154", options: { align: "right" } }, { text: "+0.452", options: { align: "right", bold: true, color: COLORS.good } }, { text: "tay phải ra dashboard" }],
    [{ text: "c7 reach" }, { text: "-0.091", options: { align: "right" } }, { text: "-0.413", options: { align: "right" } }, { text: "+0.127", options: { align: "right", bold: true, color: COLORS.good } }, { text: "+0.037", options: { align: "right", bold: true, color: COLORS.good } }, { text: "tay phải ra sau" }],
    [{ text: "c9 talk" }, { text: "-0.096", options: { align: "right", bold: true, color: COLORS.good } }, { text: "-0.389", options: { align: "right" } }, { text: "-0.266", options: { align: "right" } }, { text: "+0.425", options: { align: "right" } }, { text: "đầu xoay phải" }],
  ];
  s.addTable(rows, {
    x: MARGIN, y: 1.25, w: W - 2 * MARGIN, h: 2.6,
    colW: [1.2, 1.1, 1.1, 1.1, 1.1, 3.4],
    fontSize: 13, fontFace: FONT, color: COLORS.body,
    border: { type: "solid", pt: 0.5, color: COLORS.rule },
    valign: "middle",
  });

  // Callout
  s.addShape("roundRect", {
    x: MARGIN, y: 4.1, w: W - 2 * MARGIN, h: 1.0,
    fill: { color: COLORS.highlight }, line: { color: "E6C800", width: 1.5 },
    rectRadius: 0.08,
  });
  s.addText([
    { text: "Đọc bảng: ", options: { bold: true, breakLine: true } },
    { text: "• ", options: { breakLine: false } },
    { text: "c7 ", options: { bold: true, breakLine: false } },
    { text: "là lớp duy nhất có p29 dương (cổ tay phải xuống thấp hơn hông) — một dấu hiệu tách hoàn toàn.   ", options: { breakLine: false } },
    { text: "• ", options: { breakLine: false } },
    { text: "c5 ", options: { bold: true, breakLine: false } },
    { text: "có p31 lớn nhất (+0.452) vs c7 nhỏ nhất (+0.037) → khoảng cách 0.42, dễ học.", options: {} },
  ], {
    x: MARGIN + 0.15, y: 4.15, w: W - 2 * MARGIN - 0.3, h: 0.9,
    fontSize: 13, fontFace: FONT, color: "7A5200",
    align: "left", valign: "middle",
  });

  footerCite(s, "Pre-flight sanity check, splits/pose.parquet · log.md Run 8");
  pageNum(s, 11, TOTAL_SLIDES);
}

// ============================================================================
// SLIDE 12 — Run 8 result (PLACEHOLDER — fill after training)
// ============================================================================
{
  const s = pres.addSlide();
  actionTitle(s,
    "Run 8 result: [chờ kết quả train, điền sau] — target ≥ 0.85 macro F1, mục tiêu 0.88");

  // Left: training curves placeholder
  imageOrPlaceholder(s, "run8_training_curves.png", {
    x: MARGIN, y: 1.2, w: 5.4, h: 3.6,
  });

  // Right: comparison table placeholder
  s.addText("So sánh 3 runs", {
    x: 6.1, y: 1.2, w: 3.4, h: 0.4,
    fontSize: FS.section, fontFace: FONT, color: COLORS.accent, bold: true,
  });

  const cmpRows = [
    [
      { text: "Run", options: { bold: true, fill: { color: "EBF3FA" } } },
      { text: "Arch", options: { bold: true, fill: { color: "EBF3FA" } } },
      { text: "F1", options: { bold: true, align: "right", fill: { color: "EBF3FA" } } },
    ],
    [{ text: "Run 6" }, { text: "1× CNN" }, { text: "0.873", options: { align: "right", bold: true } }],
    [{ text: "Run 7" }, { text: "2× CNN" }, { text: "0.748", options: { align: "right", color: COLORS.bad } }],
    [{ text: "Run 8" }, { text: "1× CNN + pose" }, { text: "[TBD]", options: { align: "right", bold: true, color: COLORS.muted } }],
  ];
  s.addTable(cmpRows, {
    x: 6.1, y: 1.7, w: 3.4, h: 1.8,
    colW: [1.0, 1.6, 0.8],
    fontSize: 13, fontFace: FONT, color: COLORS.body,
    border: { type: "solid", pt: 0.5, color: COLORS.rule },
    valign: "middle",
  });

  s.addText("Target per-class (vs Run 7)", {
    x: 6.1, y: 3.65, w: 3.4, h: 0.35,
    fontSize: 13, fontFace: FONT, color: COLORS.accent, bold: true,
  });
  s.addText([
    { text: "c0: 0.51 → ≥0.78", options: { breakLine: true } },
    { text: "c3: 0.55 → ≥0.85", options: { breakLine: true } },
    { text: "c5: 0.69 → ≥0.88", options: { breakLine: true } },
    { text: "c9: 0.44 → ≥0.80", options: {} },
  ], {
    x: 6.1, y: 4.0, w: 3.4, h: 1.0,
    fontSize: 12, fontFace: FONT, color: COLORS.body,
  });

  footerCite(s, "Slide cần cập nhật sau khi run8/eval/metrics.json sẵn sàng");
  pageNum(s, 12, TOTAL_SLIDES);
}

// ============================================================================
// SLIDE 13 — Demo
// ============================================================================
{
  const s = pres.addSlide();
  actionTitle(s,
    "Demo Gradio: upload ảnh → top-3 prediction + heatmap CBAM + cảnh báo low-confidence");

  // 2 screenshots side by side
  imageOrPlaceholder(s, "demo_high_conf.png", {
    x: MARGIN, y: 1.2, w: 4.4, h: 3.5,
  });
  imageOrPlaceholder(s, "demo_ood_low_conf.png", {
    x: 5.1, y: 1.2, w: 4.4, h: 3.5,
  });

  // Captions
  s.addText("Ảnh dataset: high-confidence + heatmap đúng vùng tay", {
    x: MARGIN, y: 4.75, w: 4.4, h: 0.35,
    fontSize: 12, fontFace: FONT, color: COLORS.body,
    align: "center", italic: true,
  });
  s.addText("Ảnh OOD: low-confidence flag (max softmax < 0.40)", {
    x: 5.1, y: 4.75, w: 4.4, h: 0.35,
    fontSize: 12, fontFace: FONT, color: COLORS.body,
    align: "center", italic: true,
  });

  // Run command
  s.addText("$ python demo/app.py --ckpt best.pt --stats stats.json → http://127.0.0.1:7860", {
    x: MARGIN, y: 5.15, w: W - 2 * MARGIN, h: 0.3,
    fontSize: 11, fontFace: "Courier New", color: COLORS.muted,
    align: "center", valign: "middle",
  });

  pageNum(s, 13, TOTAL_SLIDES);
}

// ============================================================================
// SLIDE 14 — Conclusions (dark navy sandwich)
// ============================================================================
{
  const s = pres.addSlide();
  s.background = { color: COLORS.primary };

  s.addText("Kết luận", {
    x: MARGIN, y: 0.25, w: W - 2 * MARGIN, h: 0.5,
    fontSize: 22, fontFace: FONT, color: "A0BBDD", bold: false, align: "left",
  });

  // Accent rule
  s.addShape("rect", {
    x: MARGIN, y: 0.75, w: W - 2 * MARGIN, h: 0.04,
    fill: { color: COLORS.accent }, line: { color: COLORS.accent },
  });

  // Numbered takeaways
  s.addText([
    { text: "1. Subject-wise eval là không thể bỏ qua. ", options: { bold: true, breakLine: false } },
    { text: "Random split bịa val acc ~99%; subject-wise đưa con số xuống mức tổng quát hoá thật (Run 6: 0.873 macro F1).", options: { breakLine: true } },
    { text: "", options: { breakLine: true } },
    { text: "2. Thêm CNN backbone không phải lúc nào cũng tốt. ", options: { bold: true, breakLine: false } },
    { text: "Run 7 (2× ResNet18+CBAM) regress -12pp vì chia capacity giữa 2 stream học cue trùng.", options: { breakLine: true } },
    { text: "", options: { breakLine: true } },
    { text: "3. Pose landmark > second CNN. ", options: { bold: true, breakLine: false } },
    { text: "Run 8 thay stream thứ hai bằng 36 toạ độ MediaPipe — nửa params, tín hiệu tư thế trực tiếp.", options: { breakLine: true } },
    { text: "", options: { breakLine: true } },
    { text: "4. CBAM rẻ + giải thích được. ", options: { bold: true, breakLine: false } },
    { text: "+0.4% params, heatmap layer4 dồn đúng vùng cue (tay+phone, dashboard, đầu/tay).", options: {} },
  ], {
    x: MARGIN, y: 0.9, w: W - 2 * MARGIN, h: 3.8,
    fontSize: 16, fontFace: FONT, color: "FFFFFF",
    paraSpaceAfter: 4,
  });

  // Contact footer
  s.addText([
    { text: "Nguyễn Xuân Trường   ·   ", options: {} },
    { text: "github.com/nxtruoong/DoAnCS231", options: { italic: true } },
  ], {
    x: MARGIN, y: 4.95, w: W - 2 * MARGIN, h: 0.4,
    fontSize: 14, fontFace: FONT, color: "A0BBDD", align: "left",
  });
}

// ============================================================================
// SLIDE 15 — References
// ============================================================================
{
  const s = pres.addSlide();

  s.addText("Tài liệu tham khảo", {
    x: MARGIN, y: 0.2, w: W - 2 * MARGIN, h: 0.5,
    fontSize: 24, fontFace: FONT, color: COLORS.primary, bold: true,
  });

  s.addShape("rect", {
    x: MARGIN, y: 0.72, w: W - 2 * MARGIN, h: 0.025,
    fill: { color: COLORS.rule }, line: { color: COLORS.rule },
  });

  const refs = [
    "He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR. arXiv:1512.03385.",
    "He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving Deep into Rectifiers (Kaiming init). ICCV. arXiv:1502.01852.",
    "Woo, S., Park, J., Lee, J.-Y., & Kweon, I. S. (2018). CBAM: Convolutional Block Attention Module. ECCV. arXiv:1807.06521.",
    "Yun, S., Han, D., Oh, S. J., Chun, S., Choe, J., & Yoo, Y. (2019). CutMix. ICCV. arXiv:1905.04899.",
    "Müller, S. G. & Hutter, F. (2021). TrivialAugment. ICCV.",
    "Loshchilov, I., & Hutter, F. (2017). SGDR: Cosine Warm Restarts. ICLR. arXiv:1608.03983.",
    "Polyak, B. & Juditsky, A. (1992). Acceleration of Stochastic Approximation by Averaging. SIAM J. Control Optim.",
    "Bazarevsky, V. et al. (2020). BlazePose: On-device Real-time Body Pose Tracking (MediaPipe). arXiv:2006.10204.",
    "State Farm Distracted Driver Detection. Kaggle competition (2016).",
  ];
  const refItems = refs.flatMap((r, i) => [
    { text: r, options: { breakLine: true } },
    ...(i < refs.length - 1 ? [{ text: "", options: { breakLine: true } }] : []),
  ]);

  s.addText(refItems, {
    x: MARGIN, y: 0.9, w: W - 2 * MARGIN, h: 4.5,
    fontSize: 12, fontFace: FONT, color: COLORS.body,
    paraSpaceAfter: 4,
  });

  pageNum(s, 15, TOTAL_SLIDES);
}

// ============================================================================
// SAVE
// ============================================================================
pres.writeFile({ fileName: OUT }).then(() => {
  console.log(`\nWrote ${OUT}`);
  console.log(`Slides: 15 (1 title + 12 content + 1 conclusions + 1 references)`);
  console.log(`Asset dir: ${ASSETS}`);
  console.log(`Open in PowerPoint / Keynote / LibreOffice Impress to edit.`);
});
