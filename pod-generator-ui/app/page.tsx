"use client";

import { useEffect, useMemo, useState } from "react";

type GenerateResponse = {
  job_id?: string;
};

type StatusResponse = {
  job_id?: string;
  status?: string; // queued | running | done | error | unknown
  step?: string;
  progress?: number; // 0..100
  message?: string;
  error?: string | null;
};

const API_BASE = "http://127.0.0.1:8000";

type ImageItem = string | { url: string; label?: string };

const isImageObj = (x: any): x is { url: string; label?: string } =>
  x && typeof x === "object" && typeof x.url === "string";

const normalizeImages = (arr: any): string[] => {
  if (!Array.isArray(arr)) return [];
  // Convert both formats into a clean string[]
  return arr
    .map((x) => (typeof x === "string" ? x : isImageObj(x) ? x.url : null))
    .filter((x): x is string => typeof x === "string" && x.length > 0);
};

export default function Home() {
  const [prompt, setPrompt] = useState("");
  const [count, setCount] = useState(3);

  const [loading, setLoading] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);

  const [images, setImages] = useState<string[]>([]);
  const [zipUrl, setZipUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Progress
  const [status, setStatus] = useState<string>("idle");
  const [step, setStep] = useState<string>("");
  const [progress, setProgress] = useState<number>(0);
  const [message, setMessage] = useState<string>("");

  const canGenerate = useMemo(
    () => prompt.trim().length > 2 && !loading,
    [prompt, loading]
  );

  const absolutize = (pathOrUrl: unknown) => {
    if (typeof pathOrUrl !== "string") return "";
    if (!pathOrUrl) return pathOrUrl;
    if (pathOrUrl.startsWith("http://") || pathOrUrl.startsWith("https://")) return pathOrUrl;
    return `${API_BASE}${pathOrUrl}`;
  };

  const labelFor = (pathOrUrl: string) => {
    if (!pathOrUrl) return "Image";

    const clean = pathOrUrl.split("?")[0];
    const base = clean.split("/").pop()?.toLowerCase() ?? "";

    // Print files
    if (base === "print_dark.png" || base === "for_dark_shirts.png") return "Print - Dark Shirts";
    if (base === "print_light.png" || base === "for_light_shirts.png") return "Print - Light Shirts";

    // Mockups
    if (base === "mockup_black.png") return "Mockup - Black Shirt";
    if (base === "mockup_blue.png") return "Mockup - Blue Shirt";
    if (base === "mockup_white.png") return "Mockup - White Shirt";

    // Legacy / internal (if ever shown)
    if (base.startsWith("preview_print_")) return "Preview Print (Internal)";
    if (base.includes("01_original")) return "Original";
    if (base.includes("02_upscaled")) return "Upscaled";
    if (base.includes("03_transparent")) return "Transparent Cutout";

    // Fallback
    if (base.startsWith("mockup_")) return "Mockup";
    if (base.startsWith("print_")) return "Print File";

    return base || "Image";
  };

  async function fetchJob(job_id: string) {
    const res = await fetch(`${API_BASE}/job/${job_id}`, { cache: "no-store" });
    if (!res.ok) {
      const txt = await res.text().catch(() => "");
      throw new Error(`Job fetch failed (${res.status}) ${txt}`);
    }
    const data = await res.json();

    setImages(normalizeImages(data.images));
    setZipUrl(data.zip ? data.zip : null);
  }

  // Poll status while running
  useEffect(() => {
    if (!jobId) return;

    let timer: any = null;
    let stopped = false;

    const poll = async () => {
      try {
        const res = await fetch(`${API_BASE}/job/${jobId}/status`, { cache: "no-store" });
        if (!res.ok) {
          timer = setTimeout(poll, 2000);
          return;
        }

        const s: StatusResponse = await res.json();
        if (stopped) return;

        const nextStatus = s.status ?? "unknown";
        const nextStep = s.step ?? "";
        const nextProgress = typeof s.progress === "number" ? s.progress : 0;
        const nextMessage = s.message ?? "";

        setStatus(nextStatus);
        setStep(nextStep);
        setProgress(nextProgress);
        setMessage(nextMessage);

        if (nextStatus === "done") {
          await fetchJob(jobId);
          setLoading(false);
          return;
        }

        if (nextStatus === "error") {
          setLoading(false);
          setError(s.error ?? "Job failed");
          return;
        }

        timer = setTimeout(poll, 1500);
      } catch {
        timer = setTimeout(poll, 2000);
      }
    };

    poll();

    return () => {
      stopped = true;
      if (timer) clearTimeout(timer);
    };
  }, [jobId]);

  const generateDesigns = async () => {
    setError(null);
    setImages([]);
    setZipUrl(null);

    // reset progress
    setStatus("queued");
    setStep("queued");
    setProgress(0);
    setMessage("Queued...");

    setLoading(true);
    setJobId(null);

    try {
      const res = await fetch(`${API_BASE}/generate-pod-design`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, num_designs: count }),
      });

      if (!res.ok) {
        const txt = await res.text();
        throw new Error(`Generate failed (${res.status}): ${txt}`);
      }

      const data: GenerateResponse = await res.json();
      const jid = data.job_id ?? null;

      if (!jid) throw new Error("Backend did not return job_id");

      setJobId(jid);

      // IMPORTANT: do NOT setLoading(false) here.
      // We stay loading until status says done/error.
    } catch (e: any) {
      setLoading(false);
      setError(e?.message ?? "Unknown error");
      setStatus("error");
      setMessage("Failed");
      setProgress(100);
    }
  };

  const showProgressCard =
    loading || (jobId && status !== "idle" && status !== "done" && status !== "error");

  return (
    <div className="page">
      <div className="card">
        <h1 className="title">POD Design Generator</h1>
        <p className="subtitle">Generate multiple print-ready designs and mockups</p>

        <label style={{ fontWeight: 700, opacity: 0.9 }}>Prompt</label>
        <textarea
          className="textarea"
          placeholder='Example: "Cute astronaut panda, bold outlines, transparent background"'
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />

        <div className="sliderRow">
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6, opacity: 0.9 }}>
            <span style={{ fontWeight: 700 }}>Variations</span>
            <span>{count}</span>
          </div>
          <input
            type="range"
            min={1}
            max={10}
            value={count}
            onChange={(e) => setCount(Number(e.target.value))}
          />
        </div>

        <button className="generateBtn" disabled={!canGenerate} onClick={generateDesigns}>
          {loading ? "Working..." : "Generate Designs"}
        </button>

        {showProgressCard && (
          <div
            style={{
              marginTop: 16,
              padding: 14,
              borderRadius: 14,
              border: "1px solid rgba(255,255,255,0.12)",
              background: "rgba(255,255,255,0.06)",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, opacity: 0.9 }}>
              <span>{message || "Working..."}</span>
              <span>{Math.max(0, Math.min(100, progress))}%</span>
            </div>

            <div
              style={{
                marginTop: 10,
                height: 8,
                borderRadius: 999,
                background: "rgba(255,255,255,0.12)",
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  height: 8,
                  width: `${Math.max(0, Math.min(100, progress))}%`,
                  borderRadius: 999,
                  background: "linear-gradient(90deg, rgba(99,102,241,0.95), rgba(236,72,153,0.95))",
                  transition: "width 500ms ease",
                }}
              />
            </div>

            <div style={{ marginTop: 8, fontSize: 12, opacity: 0.65 }}>
              Step: {step || "starting"} {jobId ? `| Job: ${jobId}` : ""}
            </div>
          </div>
        )}

        {error && (
          <div style={{ marginTop: 16, padding: 12, borderRadius: 12, background: "rgba(239,68,68,0.15)" }}>
            <div style={{ color: "#fecaca", fontSize: 14 }}>{error}</div>
          </div>
        )}

        {!loading && (zipUrl || images.length > 0) && (
          <div className="resultCard">
            <div style={{ fontWeight: 800 }}>Designs Ready!</div>

            {zipUrl && (
              <a href={absolutize(zipUrl)} target="_blank" rel="noreferrer">
                Download ZIP
              </a>
            )}
          </div>
        )}

        <div style={{ marginTop: 22 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
            <h2 style={{ fontSize: 22, fontWeight: 900 }}>Gallery</h2>
            <div style={{ opacity: 0.65, fontSize: 14 }}>
              {images.length > 0 ? `${images.length} images` : "No designs yet"}
            </div>
          </div>

          {images.length === 0 ? (
            <div style={{ marginTop: 10, opacity: 0.7, fontSize: 14 }}>
              Generate designs to preview them here.
            </div>
          ) : (
            <div
              style={{
                marginTop: 14,
                display: "grid",
                gridTemplateColumns: "repeat(2, 1fr)",
                gap: 12,
              }}
            >
              {images.map((img, idx) => (
                <a
                  key={`${img}-${idx}`}
                  href={absolutize(img)}
                  target="_blank"
                  rel="noreferrer"
                  style={{
                    borderRadius: 16,
                    overflow: "hidden",
                    border: "1px solid rgba(255,255,255,0.12)",
                    background: "rgba(255,255,255,0.06)",
                    boxShadow: "0 10px 40px rgba(0,0,0,0.35)",
                  }}
                >
                  <img
                    src={absolutize(img)}
                    alt={labelFor(img)}
                    style={{ width: "100%", height: 220, objectFit: "cover", display: "block" }}
                    loading="lazy"
                  />
                  <div style={{ padding: 10, fontSize: 12, opacity: 0.85 }}>
                    {labelFor(img)}
                  </div>
                </a>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
