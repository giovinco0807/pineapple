import fs from "node:fs/promises";
import { createReadStream } from "node:fs";
import readline from "node:readline";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const BASE_DIR = "D:/ofc-pineapple-data/branch_expanded_t0top50_t1t2top10_20260605";
const OUTPUT_DIR = `${BASE_DIR}/reports`;
const OUTPUT_XLSX = `${OUTPUT_DIR}/branch_deal_distribution_audit.xlsx`;
const PREVIEW_DIR = `${OUTPUT_DIR}/_previews`;

const INPUTS = [
  {
    label: "BB root1",
    position: "bb",
    input: `${BASE_DIR}/inputs/bb_root1_t3_inputs.jsonl`,
    exact: `${BASE_DIR}/exact/bb_root1_t3_exact.rust.jsonl`,
    teacher: `${BASE_DIR}/teacher/bb_root1_t3_teacher.jsonl`,
    reranker: `${BASE_DIR}/reranker/bb_root1_t3_exact`,
  },
  {
    label: "BTN root1",
    position: "btn",
    input: `${BASE_DIR}/inputs/btn_root1_t3_inputs.jsonl`,
    exact: `${BASE_DIR}/exact/btn_root1_t3_exact.rust.jsonl`,
    teacher: `${BASE_DIR}/teacher/btn_root1_t3_teacher.jsonl`,
    reranker: `${BASE_DIR}/reranker/btn_root1_t3_exact`,
  },
];

const COMBINED_RERANKER = `${BASE_DIR}/reranker/bb_btn_root1_t3_exact`;
const RANKS = "23456789TJQKA";

function colName(index) {
  let n = index + 1;
  let out = "";
  while (n > 0) {
    const rem = (n - 1) % 26;
    out = String.fromCharCode(65 + rem) + out;
    n = Math.floor((n - 1) / 26);
  }
  return out;
}

function rangeFor(startRow, startCol, rows) {
  const endRow = startRow + rows.length - 1;
  const endCol = startCol + rows[0].length - 1;
  return `${colName(startCol)}${startRow}:${colName(endCol)}${endRow}`;
}

function cardRank(card) {
  if (!card) return "";
  if (String(card).startsWith("X")) return "X";
  return String(card)[0];
}

function cardSuit(card) {
  if (!card) return "";
  if (String(card).startsWith("X")) return "Joker";
  return String(card).slice(1);
}

function addCount(map, key, amount = 1) {
  map.set(key, (map.get(key) || 0) + amount);
}

function boardText(board, row) {
  return ((board || {})[row] || []).join(" ");
}

function stableDeal(cards) {
  return (cards || []).join(" ");
}

async function fileLines(path) {
  try {
    const stream = createReadStream(path, { encoding: "utf8" });
    const rl = readline.createInterface({ input: stream, crlfDelay: Infinity });
    let n = 0;
    for await (const line of rl) {
      if (line.trim()) n += 1;
    }
    return n;
  } catch {
    return 0;
  }
}

async function fileSize(path) {
  try {
    const stat = await fs.stat(path);
    return stat.size;
  } catch {
    return 0;
  }
}

async function readJson(path) {
  try {
    return JSON.parse(await fs.readFile(path, "utf8"));
  } catch {
    return {};
  }
}

async function analyzeInput(source) {
  const dealCounts = new Map();
  const targetDealCounts = new Map();
  const branchRankCounts = new Map();
  const targetCardRepeated = new Map();
  const targetCardUniqueDeals = new Map();
  const allTraceCardRepeated = new Map();
  const allTraceCardUniqueDeals = new Map();
  const sampleRows = [];
  let records = 0;
  const uniqueTargetDealPresence = new Set();
  const uniqueAllTraceDealPresence = new Set();

  const stream = createReadStream(source.input, { encoding: "utf8" });
  const rl = readline.createInterface({ input: stream, crlfDelay: Infinity });
  for await (const line of rl) {
    if (!line.trim()) continue;
    const row = JSON.parse(line);
    records += 1;
    const branch = row.branch || {};

    for (const [turn, rankKey] of [
      [0, "target_t0_rank"],
      [1, "target_t1_rank"],
      [2, "target_t2_rank"],
    ]) {
      addCount(branchRankCounts, JSON.stringify([source.position, turn, branch[rankKey]]));
    }

    const targetDeals = {};
    for (const trace of row.trace || []) {
      const turn = Number(trace.turn);
      const seat = String(trace.seat || "");
      const role = String(trace.branch_role || "");
      const deal = stableDeal(trace.dealt || []);
      const key = JSON.stringify([source.position, seat, role, turn, deal]);
      addCount(dealCounts, key);
      const uniqueAllKey = `${seat}|${role}|${turn}|${deal}`;
      if (!uniqueAllTraceDealPresence.has(uniqueAllKey)) {
        uniqueAllTraceDealPresence.add(uniqueAllKey);
        for (const card of trace.dealt || []) addCount(allTraceCardUniqueDeals, card);
      }
      for (const card of trace.dealt || []) addCount(allTraceCardRepeated, card);

      if (role === "target_branch" && turn >= 0 && turn <= 2) {
        targetDeals[`t${turn}`] = deal;
        const targetKey = JSON.stringify([source.position, turn, deal]);
        addCount(targetDealCounts, targetKey);
        const uniqueTargetKey = `${turn}|${deal}`;
        if (!uniqueTargetDealPresence.has(uniqueTargetKey)) {
          uniqueTargetDealPresence.add(uniqueTargetKey);
          for (const card of trace.dealt || []) addCount(targetCardUniqueDeals, card);
        }
        for (const card of trace.dealt || []) addCount(targetCardRepeated, card);
      }
    }

    if (sampleRows.length < 120) {
      sampleRows.push([
        source.position,
        records,
        branch.branch_id || "",
        branch.target_t0_rank,
        branch.target_t1_rank,
        branch.target_t2_rank,
        targetDeals.t0 || "",
        targetDeals.t1 || "",
        targetDeals.t2 || "",
        stableDeal(row.dealt || []),
        boardText(row.board, "top"),
        boardText(row.board, "middle"),
        boardText(row.board, "bottom"),
        boardText(row.opponent_board, "top"),
        boardText(row.opponent_board, "middle"),
        boardText(row.opponent_board, "bottom"),
      ]);
    }
  }

  return {
    source,
    records,
    dealCounts,
    targetDealCounts,
    branchRankCounts,
    targetCardRepeated,
    targetCardUniqueDeals,
    allTraceCardRepeated,
    allTraceCardUniqueDeals,
    sampleRows,
  };
}

function mapRows(map, parseKey, valueName = "count") {
  const out = [];
  for (const [key, count] of map.entries()) {
    out.push([...parseKey(JSON.parse(key)), count]);
  }
  return out;
}

function sortedRows(rows, cols) {
  return rows.sort((a, b) => {
    for (const col of cols) {
      if (a[col] < b[col]) return -1;
      if (a[col] > b[col]) return 1;
    }
    return 0;
  });
}

function rankOrder(rank) {
  if (rank === "X") return 13;
  return RANKS.indexOf(rank);
}

function cardCountRows(label, repeated, uniqueDeals) {
  const cards = new Set([...repeated.keys(), ...uniqueDeals.keys()]);
  return [...cards].sort((a, b) => {
    const ra = rankOrder(cardRank(a));
    const rb = rankOrder(cardRank(b));
    if (ra !== rb) return ra - rb;
    return String(a).localeCompare(String(b));
  }).map((card) => [
    label,
    card,
    cardRank(card),
    cardSuit(card),
    repeated.get(card) || 0,
    uniqueDeals.get(card) || 0,
  ]);
}

function rankCountRows(label, repeated, uniqueDeals) {
  const repeatedRank = new Map();
  const uniqueRank = new Map();
  for (const [card, count] of repeated.entries()) addCount(repeatedRank, cardRank(card), count);
  for (const [card, count] of uniqueDeals.entries()) addCount(uniqueRank, cardRank(card), count);
  const ranks = new Set([...repeatedRank.keys(), ...uniqueRank.keys()]);
  return [...ranks].sort((a, b) => rankOrder(a) - rankOrder(b)).map((rank) => [
    label,
    rank,
    repeatedRank.get(rank) || 0,
    uniqueRank.get(rank) || 0,
  ]);
}

function addSheet(workbook, name, rows, tableName, widths = []) {
  const sheet = workbook.worksheets.add(name);
  sheet.showGridLines = false;
  if (!rows.length || !rows[0].length) return sheet;
  const range = rangeFor(1, 0, rows);
  sheet.getRange(range).values = rows;
  sheet.getRange(`${colName(0)}1:${colName(rows[0].length - 1)}1`).format = {
    fill: "#1F2937",
    font: { bold: true, color: "#FFFFFF" },
  };
  if (rows.length > 1) {
    const table = sheet.tables.add(range, true, tableName);
    table.showFilterButton = true;
    table.showBandedRows = true;
  }
  sheet.freezePanes.freezeRows(1);
  for (let i = 0; i < rows[0].length; i += 1) {
    const width = widths[i] || 120;
    sheet.getRange(`${colName(i)}1:${colName(i)}${Math.max(rows.length, 2)}`).format.columnWidthPx = width;
  }
  return sheet;
}

function addTitleSheet(workbook, summaryRows, coverageRows) {
  const sheet = workbook.worksheets.add("Summary");
  sheet.showGridLines = false;
  sheet.getRange("A1").values = [["Branch-expanded T3 deal distribution audit"]];
  sheet.mergeCells("A1:H1");
  sheet.getRange("A1:H1").format = {
    fill: "#111827",
    font: { bold: true, color: "#FFFFFF", size: 14 },
  };
  sheet.getRange("A3:B12").values = summaryRows;
  sheet.getRange("A3:B3").format = {
    fill: "#374151",
    font: { bold: true, color: "#FFFFFF" },
  };
  sheet.getRange("A14:G14").values = [coverageRows[0]];
  sheet.getRange("A15:G" + (13 + coverageRows.length)).values = coverageRows.slice(1);
  sheet.getRange("A14:G14").format = {
    fill: "#374151",
    font: { bold: true, color: "#FFFFFF" },
  };
  const widths = [330, 500, 210, 130, 170, 210, 210, 30];
  for (let i = 0; i < widths.length; i += 1) {
    sheet.getRange(`${colName(i)}1:${colName(i)}40`).format.columnWidthPx = widths[i];
  }
  sheet.freezePanes.freezeRows(2);
  return sheet;
}

async function main() {
  await fs.mkdir(OUTPUT_DIR, { recursive: true });
  await fs.mkdir(PREVIEW_DIR, { recursive: true });

  const analyses = [];
  for (const input of INPUTS) analyses.push(await analyzeInput(input));

  const combinedMeta = await readJson(`${COMBINED_RERANKER}/metadata.json`);
  const sourceRows = [[
    "label",
    "position",
    "input_path",
    "input_records",
    "input_bytes",
    "exact_path",
    "exact_records",
    "teacher_path",
    "teacher_records",
    "reranker_dir",
    "reranker_samples",
  ]];
  for (const analysis of analyses) {
    const src = analysis.source;
    const meta = await readJson(`${src.reranker}/metadata.json`);
    sourceRows.push([
      src.label,
      src.position,
      src.input,
      analysis.records,
      await fileSize(src.input),
      src.exact,
      await fileLines(src.exact),
      src.teacher,
      await fileLines(src.teacher),
      src.reranker,
      meta.n_samples || 0,
    ]);
  }

  const dealCoverageRows = [[
    "position",
    "target_turn",
    "unique_target_deal_combos",
    "repeated_rows",
    "rows_per_unique_deal",
    "unique_cards_in_target_deals",
    "unique_ranks_in_target_deals",
  ]];
  for (const analysis of analyses) {
    for (const turn of [0, 1, 2]) {
      const rows = mapRows(
        analysis.targetDealCounts,
        ([position, t, deal]) => [position, t, deal],
      ).filter((r) => r[0] === analysis.source.position && r[1] === turn);
      const repeated = rows.reduce((sum, r) => sum + r[3], 0);
      const combos = new Set(rows.map((r) => r[2]));
      const cards = new Set();
      const ranks = new Set();
      for (const deal of combos) {
        for (const card of String(deal).split(" ").filter(Boolean)) {
          cards.add(card);
          ranks.add(cardRank(card));
        }
      }
      dealCoverageRows.push([
        analysis.source.position,
        turn,
        combos.size,
        repeated,
        combos.size ? repeated / combos.size : 0,
        cards.size,
        ranks.size,
      ]);
    }
  }

  const allTurnDealRows = [[
    "generated_position",
    "acting_seat",
    "role",
    "turn",
    "dealt_combo",
    "occurrences",
  ]];
  for (const analysis of analyses) {
    allTurnDealRows.push(
      ...sortedRows(
        mapRows(analysis.dealCounts, ([position, seat, role, turn, deal]) => [position, seat, role, turn, deal]),
        [0, 1, 3, 2, 4],
      ),
    );
  }

  const rankRows = [["position", "turn", "target_model_rank", "occurrences"]];
  for (const analysis of analyses) {
    rankRows.push(
      ...sortedRows(
        mapRows(analysis.branchRankCounts, ([position, turn, rank]) => [position, turn, rank]),
        [0, 1, 2],
      ),
    );
  }

  const cardRows = [["scope", "card", "rank", "suit", "repeated_deal_appearances", "unique_deal_presence"]];
  const rankCountSummaryRows = [["scope", "rank", "repeated_deal_appearances", "unique_deal_presence"]];
  for (const analysis of analyses) {
    cardRows.push(
      ...cardCountRows(`${analysis.source.position}_target_t0_t2`, analysis.targetCardRepeated, analysis.targetCardUniqueDeals),
      ...cardCountRows(`${analysis.source.position}_all_trace`, analysis.allTraceCardRepeated, analysis.allTraceCardUniqueDeals),
    );
    rankCountSummaryRows.push(
      ...rankCountRows(`${analysis.source.position}_target_t0_t2`, analysis.targetCardRepeated, analysis.targetCardUniqueDeals),
      ...rankCountRows(`${analysis.source.position}_all_trace`, analysis.allTraceCardRepeated, analysis.allTraceCardUniqueDeals),
    );
  }

  const sampleRows = [[
    "position",
    "input_row",
    "branch_id",
    "t0_rank",
    "t1_rank",
    "t2_rank",
    "target_t0_dealt",
    "target_t1_dealt",
    "target_t2_dealt",
    "target_t3_dealt",
    "board_top",
    "board_middle",
    "board_bottom",
    "opponent_top",
    "opponent_middle",
    "opponent_bottom",
  ]];
  for (const analysis of analyses) sampleRows.push(...analysis.sampleRows);

  const totalInputRows = analyses.reduce((sum, a) => sum + a.records, 0);
  const summaryRows = [
    ["metric", "value"],
    ["important_finding", "Only one unique target deal combo per position/turn in this pilot"],
    ["why_it_is_biased", "The branch expansion diversifies placements, but T0/T1/T2 dealt cards are fixed for each root"],
    ["total_t3_positions", totalInputRows],
    ["combined_reranker_candidate_samples", combinedMeta.n_samples || 0],
    ["combined_reranker_groups", combinedMeta.n_records || 0],
    ["bb_positions", analyses.find((a) => a.source.position === "bb")?.records || 0],
    ["btn_positions", analyses.find((a) => a.source.position === "btn")?.records || 0],
    ["local_generation_note", "More root decks are needed before training conclusions"],
    ["rough_time_for_this_pilot", "about 27 minutes for BB+BTN root1 exact labeling"],
  ];

  const workbook = Workbook.create();
  addTitleSheet(workbook, summaryRows, dealCoverageRows);
  addSheet(workbook, "TurnDeals", allTurnDealRows, "TurnDealsTable", [120, 110, 140, 80, 180, 110]);
  addSheet(workbook, "BranchRanks", rankRows, "BranchRanksTable", [110, 80, 140, 110]);
  addSheet(workbook, "CardCounts", cardRows, "CardCountsTable", [180, 80, 70, 80, 170, 150]);
  addSheet(workbook, "RankCounts", rankCountSummaryRows, "RankCountsTable", [180, 80, 170, 150]);
  addSheet(workbook, "SampleBranches", sampleRows, "SampleBranchesTable", [90, 90, 110, 80, 80, 80, 150, 150, 150, 150, 150, 180, 180, 150, 180, 180]);
  addSheet(workbook, "SourceFiles", sourceRows, "SourceFilesTable", [110, 80, 420, 110, 110, 420, 110, 420, 110, 420, 120]);

  const summaryInspect = await workbook.inspect({
    kind: "table",
    range: "Summary!A1:H25",
    include: "values,formulas",
    tableMaxRows: 25,
    tableMaxCols: 8,
  });
  console.log(summaryInspect.ndjson);
  const errors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 50 },
    summary: "formula error scan",
  });
  console.log(errors.ndjson);

  for (const name of ["Summary", "TurnDeals", "BranchRanks", "CardCounts", "SampleBranches", "SourceFiles"]) {
    const preview = await workbook.render({ sheetName: name, autoCrop: "all", scale: 1, format: "png" });
    await fs.writeFile(`${PREVIEW_DIR}/${name}.png`, new Uint8Array(await preview.arrayBuffer()));
  }

  const xlsx = await SpreadsheetFile.exportXlsx(workbook);
  await xlsx.save(OUTPUT_XLSX);
  console.log(JSON.stringify({ output: OUTPUT_XLSX, totalInputRows, combinedSamples: combinedMeta.n_samples || 0 }, null, 2));
}

await main();
