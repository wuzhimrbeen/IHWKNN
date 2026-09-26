import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const base = "E:/论文/BMC_Bioinformatics_submission/revision_round1_20260916/experiments/final_case_study";
const out = path.join(base, "outputs");
const datasets = ["Cdataset", "Fdataset", "iDrug", "LAGCN", "LRSSL", "Ydataset"];
const priorPayload = JSON.parse(await fs.readFile(path.join(out, "prior_case_study_matches.json"), "utf8"));
const priorMap = new Map(priorPayload.matches.map(row => [`${row.dataset}\t${row.disease_id}\t${row.drug_id}`, row]));
const candidates = [];
const audits = [];
for (const dataset of datasets) {
  const dir = path.join(out, dataset);
  const rows = JSON.parse(await fs.readFile(path.join(dir, "top15.json"), "utf8"));
  const audit = JSON.parse(await fs.readFile(path.join(dir, "audit.json"), "utf8"));
  if (rows.length !== 15 || rows.some((row, index) => row.Dataset !== dataset || row.Rank !== index + 1 || row["Verification status"] !== "Not checked")) {
    throw new Error(`Invalid top-15 input for ${dataset}`);
  }
  candidates.push(...rows);
  audits.push(audit);
}
if (candidates.length !== 90) throw new Error("Expected 90 rows");

const wb = Workbook.create();
const sheet = wb.worksheets.add("Top15 candidates");
sheet.showGridLines = false;
sheet.getRange("A1:Q1").merge();
sheet.getRange("A1").values = [["Final IHWKNN · Top-15 unobserved pairs per dataset"]];
sheet.getRange("A2:Q2").merge();
sheet.getRange("A2").values = [["For manual verification only. Scores are ranking values, not probabilities or evidence of confirmed associations."]];
sheet.getRange("A3:Q3").merge();
sheet.getRange("A3").values = [["Six datasets included; SCMFDDL and TLHGBI excluded because their identifiers cannot be mapped reliably for this case study."]];
const headers = ["Dataset", "Rank", "Disease ID", "Drug ID", "Drug name", "Disease name", "IHWKNN score", "Disease URL", "Drug URL", "Evidence URL / PMID", "Verification status", "Notes", "Drug row (0-based)", "Disease column (0-based)", "Prior rank", "Prior rationale", "Prior source URLs"];
sheet.getRange("A4:Q4").values = [headers];
const values = candidates.map(row => headers.map(header => {
  const prior = priorMap.get(`${row.Dataset}\t${row["Disease ID"]}\t${row["Drug ID"]}`);
  if (header === "Verification status") return prior ? `Prior check: ${prior.prior_detailed_result}` : "Not checked";
  if (header === "Notes" && prior) {
    const conflict = prior.status_conflict ? ` Old summary workbook says ${prior.prior_summary_result}; detailed log says ${prior.prior_detailed_result}.` : "";
    return `Prior automated keyword review only; direct pair evidence still needed.${conflict}`;
  }
  if (header === "Prior rank") return prior ? prior.prior_rank : null;
  if (header === "Prior rationale") return prior ? prior.prior_rationale : "";
  if (header === "Prior source URLs") return prior ? prior.prior_sources.replaceAll(" | ", "\n") : "";
  const value = row[header];
  if ((header === "Drug name" && value === row["Drug ID"]) || (header === "Disease name" && value === row["Disease ID"])) return "";
  return value;
}));
sheet.getRange(`A5:Q${4 + values.length}`).values = values;
sheet.freezePanes.freezeRows(4);
sheet.getRange("A1:Q1").format = {fill: "#123A5A", font: {name: "Arial", size: 16, bold: true, color: "#FFFFFF"}};
sheet.getRange("A2:Q3").format = {fill: "#EAF2F8", font: {name: "Arial", size: 10, color: "#25465F"}};
sheet.getRange("A4:Q4").format = {fill: "#24628A", font: {name: "Arial", size: 10, bold: true, color: "#FFFFFF"}, wrapText: true};
sheet.getRange("A5:Q94").format.font = {name: "Arial", size: 10, color: "#1D2D3A"};
sheet.getRange("A1:Q1").format.rowHeight = 30;
sheet.getRange("A2:Q3").format.rowHeight = 22;
sheet.getRange("A4:Q4").format.rowHeight = 36;
sheet.getRange("A5:Q94").format.rowHeight = 20;
sheet.getRange("L5:L94").format.wrapText = true;
sheet.getRange("P5:Q94").format.wrapText = true;
const widths = {A: 14, B: 7, C: 13, D: 14, E: 25, F: 26, G: 15, H: 49, I: 59, J: 34, K: 22, L: 52, M: 18, N: 22, O: 12, P: 78, Q: 78};
for (const [col, width] of Object.entries(widths)) sheet.getRange(`${col}:${col}`).format.columnWidth = width;
sheet.getRange("G5:G94").setNumberFormat("0.000000");
sheet.getRange("B5:B94").setNumberFormat("0");
sheet.getRange("M5:N94").setNumberFormat("0");
sheet.getRange("O5:O94").setNumberFormat("0");
sheet.getRange("C5:F94").setNumberFormat("@");
for (let block = 0; block < 6; block++) {
  const first = 5 + block * 15;
  const last = first + 14;
  if (block % 2 === 1) sheet.getRange(`A${first}:Q${last}`).format.fill = "#F2F7FA";
  sheet.getRange(`A${first}:Q${first}`).format.borders = {top: {style: "medium", color: "#7DAAC4"}};
}
for (const prior of priorPayload.matches) {
  const datasetPosition = datasets.indexOf(prior.dataset);
  const rowNumber = 5 + datasetPosition * 15 + prior.new_rank - 1;
  sheet.getRange(`A${rowNumber}:Q${rowNumber}`).format.rowHeight = 76;
}
sheet.getRange("K5:K94").dataValidation = {rule: {type: "list", values: ["Not checked", "Prior check: yes", "Prior check: uncertain", "Supported (direct evidence)", "Not supported", "Unclear"]}};
sheet.tables.add("A4:Q94", true, "FinalCandidates");

const auditSheet = wb.worksheets.add("Run audit");
auditSheet.showGridLines = false;
auditSheet.getRange("A1:J1").merge();
auditSheet.getRange("A1").values = [["Full-data final-model ranking · protocol and provenance"]];
auditSheet.getRange("A1:J1").format = {fill: "#123A5A", font: {name: "Arial", size: 14, bold: true, color: "#FFFFFF"}};
auditSheet.getRange("A2:B7").values = [
  ["Model", "IHWKNN (final configuration)"],
  ["K", 120], ["lambda", 0.5], ["gamma", 0.1], ["beta", 0.3], ["alpha", 4],
];
auditSheet.getRange("A8:B11").values = [
  ["Training", "All known links in each dataset; no held-out positives"],
  ["Candidates", "Only original zero entries (unobserved pairs); known links excluded"],
  ["Ranking", "Descending raw score; ties resolved by matrix row-major index"],
  ["Interpretation", "Predictions, not validated treatment indications"],
];
auditSheet.getRange("A13:J13").values = [["Dataset", "Drugs", "Diseases", "Known links", "Unknown pairs", "Stop iteration", "Stop reason", "Boundary", "Final support", "Runtime (s)"]];
auditSheet.getRange("A14:J19").values = audits.map(a => [a.dataset, a.drugs, a.diseases, a.known_associations, a.unknown_pairs, a.selected_iteration, a.stop_reason, a.boundary, a.final_nonzero_support, a.runtime_seconds]);
auditSheet.getRange("A21:B23").values = [
  ["Excluded", "Reason"],
  ["SCMFDDL", "Drug identifier list has 1,323 rows but association matrix has 1,319 drug rows."],
  ["TLHGBI", "Disease identifiers unavailable for reliable database lookup."],
];
auditSheet.getRange("A25:B26").values = [
  ["Source matrices", "E:/论文/代码/01newdata/<dataset>/ANMF/"],
  ["Identifier files", "E:/论文/代码/01newdata/name/Dataset_Drug_Disease_info/<dataset>/"],
];
auditSheet.getRange("A28:B31").values = [
  ["Prior checks", `${priorPayload.matched_count} exact matches to prior case-study detailed log`],
  ["Match rule", priorPayload.matching_rule],
  ["Status scope", "Old checks were keyword-level; direct association evidence has not been established by this import."],
  ["Conflicts", "When old summary and detailed log differ, detailed log is shown and the conflict is flagged in Notes."],
];
auditSheet.getRange("A32:B33").values = [
  ["Detailed source", path.basename(priorPayload.source_detailed_csv)],
  ["Summary source", path.basename(priorPayload.source_summary_workbook)],
];
auditSheet.getRange("A2:J33").format.font = {name: "Arial", size: 10, color: "#1D2D3A"};
auditSheet.getRange("A13:J13").format = {fill: "#24628A", font: {name: "Arial", size: 10, bold: true, color: "#FFFFFF"}};
auditSheet.getRange("A21:B21").format = {fill: "#24628A", font: {name: "Arial", size: 10, bold: true, color: "#FFFFFF"}};
auditSheet.getRange("A:A").format.columnWidth = 20;
auditSheet.getRange("B:B").format.columnWidth = 76;
auditSheet.getRange("C:J").format.columnWidth = 17;
auditSheet.getRange("H14:H19").setNumberFormat("0.00");
auditSheet.getRange("J14:J19").setNumberFormat("0.000");
auditSheet.freezePanes.freezeRows(13);

wb.recalculate();
const inspect = await wb.inspect({kind: "region", sheetId: "Top15 candidates", range: "J4:Q12", maxChars: 3000});
console.log(JSON.stringify(inspect));
const errors = await wb.inspect({kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!", options: {useRegex: true, maxResults: 30}, summary: "final error scan"});
console.log(JSON.stringify(errors));
const preview = await wb.render({sheetName: "Top15 candidates", autoCrop: "all", scale: 0.7, format: "png"});
await fs.writeFile(path.join(out, "top15_with_prior_checks_preview.png"), new Uint8Array(await preview.arrayBuffer()));
const detailPreview = await wb.render({sheetName: "Top15 candidates", range: "J4:Q12", scale: 1.3, format: "png"});
await fs.writeFile(path.join(out, "prior_checks_detail_preview.png"), new Uint8Array(await detailPreview.arrayBuffer()));
const auditPreview = await wb.render({sheetName: "Run audit", autoCrop: "all", scale: 1, format: "png"});
await fs.writeFile(path.join(out, "prior_checks_audit_preview.png"), new Uint8Array(await auditPreview.arrayBuffer()));
const target = path.join(out, "IHWKNN_final_model_top15_with_prior_checks_20260919.xlsx");
const blob = await SpreadsheetFile.exportXlsx(wb);
await blob.save(target);
console.log(target);
