import fs from 'fs';
import path from 'path';


export default async function handler(req, res) {

  // Set correct CORS header if origin is allowed
  res.setHeader('Access-Control-Allow-Origin', 'https://zupstn.com');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  const filePath = path.join(process.cwd(), 'data', 'graphData.json');

  try {
    const fileContents = fs.readFileSync(filePath, 'utf8');
    const data = JSON.parse(fileContents);

    res.status(200).json(data);
  } catch (err) {
    console.error('Failed to read JSON:', err);
    res.status(500).json({ error: 'Failed to load data' });
  }
}
