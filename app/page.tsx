"use client";

import { useEffect, useRef, useState } from "react";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import "@tensorflow/tfjs";

type Prediction = {
  class: string;
  score: number;
  bbox: [number, number, number, number];
};

const CONFIDENCE_THRESHOLD = 0.6;

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const modelRef = useRef<cocoSsd.ObjectDetection | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  const [isModelLoading, setIsModelLoading] = useState(true);
  const [isWebcamReady, setIsWebcamReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [detectedObjects, setDetectedObjects] = useState<Prediction[]>([]);

  // Étape 1 : charger le modèle COCO-SSD (une seule fois, au montage du composant)
  useEffect(() => {
    cocoSsd.load().then((loadedModel) => {
      modelRef.current = loadedModel;
      setIsModelLoading(false);
    });
  }, []);

  // Étape 2 : demander l'accès à la webcam
  useEffect(() => {
    navigator.mediaDevices
      .getUserMedia({ video: true })
      .then((stream) => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            setIsWebcamReady(true);
          };
        }
      })
      .catch(() => {
        setError("Impossible d'accéder à la webcam. Vérifie les autorisations de ton navigateur.");
      });
  }, []);

  // Étape 3 : la boucle de détection, une fois modèle ET webcam prêts
  useEffect(() => {
    if (isModelLoading || !isWebcamReady) return;

    const detectFrame = async () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const model = modelRef.current;

      if (!video || !canvas || !model) return;

      // Le canvas doit avoir exactement la même résolution que la vidéo
      // pour que les coordonnées des boîtes tombent au bon endroit
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      const predictions = await model.detect(video);
      const filtered = predictions.filter((p) => p.score >= CONFIDENCE_THRESHOLD);
      setDetectedObjects(filtered);

      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        filtered.forEach((prediction) => {
          const [x, y, width, height] = prediction.bbox;

          ctx.strokeStyle = "#4ade80";
          ctx.lineWidth = 2;
          ctx.strokeRect(x, y, width, height);

          const label = `${prediction.class} (${Math.round(prediction.score * 100)}%)`;
          ctx.fillStyle = "#4ade80";
          ctx.font = "16px sans-serif";
          const textWidth = ctx.measureText(label).width;
          ctx.fillRect(x, y - 20, textWidth + 8, 20);
          ctx.fillStyle = "#0a0a0a";
          ctx.fillText(label, x + 4, y - 5);
        });
      }

      animationFrameRef.current = requestAnimationFrame(detectFrame);
    };

    detectFrame();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isModelLoading, isWebcamReady]);

  return (
    <main className="min-h-screen bg-[#0a0a0a] flex flex-col items-center justify-center gap-6 p-6">
      <div className="text-center">
        <h1 className="text-2xl font-semibold text-white mb-2">Détection d'objets en direct</h1>
        <p className="text-sm text-white/50">
          {isModelLoading
            ? "Chargement du modèle COCO-SSD..."
            : !isWebcamReady
            ? "En attente de la webcam..."
            : "Détection active"}
        </p>
      </div>

      {error && <p className="text-red-400 text-sm">{error}</p>}

      <div className="relative rounded-lg overflow-hidden border border-white/10">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="block"
          style={{ width: 640, height: 480 }}
        />
        <canvas
          ref={canvasRef}
          className="absolute top-0 left-0"
          style={{ width: 640, height: 480 }}
        />
      </div>

      {detectedObjects.length > 0 && (
        <div className="flex flex-wrap gap-2 max-w-2xl justify-center">
          {detectedObjects.map((obj, i) => (
            <span
              key={i}
              className="text-xs px-3 py-1.5 bg-white/5 border border-white/10 rounded-full text-white/70"
            >
              {obj.class} — {Math.round(obj.score * 100)}%
            </span>
          ))}
        </div>
      )}
    </main>
  );
}