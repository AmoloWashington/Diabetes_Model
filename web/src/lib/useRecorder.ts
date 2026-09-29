import { useCallback, useRef, useState } from "react";

/** Record a canvas to WebM with MediaRecorder; downloads on stop. */
export function useRecorder(getCanvas: () => HTMLCanvasElement | null | undefined, filename: string) {
  const [recording, setRecording] = useState(false);
  const rec = useRef<MediaRecorder | null>(null);

  const toggle = useCallback((): string | null => {
    if (rec.current) {
      rec.current.stop();
      return null;
    }
    const canvas = getCanvas();
    if (!canvas || !("captureStream" in canvas) || typeof MediaRecorder === "undefined") {
      return "Video recording is not supported in this browser.";
    }
    const types = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"];
    const mimeType = types.find((t) => MediaRecorder.isTypeSupported(t));
    const chunks: Blob[] = [];
    const r = new MediaRecorder(canvas.captureStream(30), mimeType ? { mimeType, videoBitsPerSecond: 6_000_000 } : undefined);
    r.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    r.onstop = () => {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(new Blob(chunks, { type: "video/webm" }));
      a.download = `${filename}-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.webm`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 4000);
      rec.current = null;
      setRecording(false);
    };
    r.start(250);
    rec.current = r;
    setRecording(true);
    return null;
  }, [getCanvas, filename]);

  return { recording, toggle };
}
