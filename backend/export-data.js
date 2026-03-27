/**
 * Exports db.js data to ai_service/data.json so the Python AI service can load it.
 * Run from project root: node backend/export-data.js
 */

const path = require("path");
const fs = require("fs");
const db = require("./db.js");

const outDir = path.join(__dirname, "..", "ai_service");
if (!fs.existsSync(outDir)) {
  fs.mkdirSync(outDir, { recursive: true });
}

const outPath = path.join(outDir, "data.json");
fs.writeFileSync(
  outPath,
  JSON.stringify(
    {
      courses: db.courses,
      users: db.users,
      interactions: db.interactions,
    },
    null,
    2,
  ),
  "utf8",
);

console.log(`Exported data to ${outPath}`);
