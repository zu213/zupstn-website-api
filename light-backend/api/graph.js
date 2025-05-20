import fetch from "node-fetch";

export default async function handler(req, res) {
  
  const data = {}

  res.setHeader("Content-Type", "image/svg+xml");
  res.setHeader("Cache-Control", "s-maxage=3600, stale-while-revalidate");
  res.send(data);
};
