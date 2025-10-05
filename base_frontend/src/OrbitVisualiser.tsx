import React, { useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Stars, Html } from "@react-three/drei";
import * as THREE from "three";
import { motion } from "framer-motion";
import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, ReferenceLine,
} from "recharts";

/* ---------- Types ---------- */
interface Detection {
  name: string;
  confidence: number;
  period_days: number;
  planet_radius_earth: number;
  transit_depth_ppm?: number;
  eq_temp_k?: number;
  folded_lightcurve?: {
    phase: number[];
    flux: number[];
  };
}

interface VisualizerProps {
  starName: string;
  detections: Detection[];
  onClose: () => void;
}

/* ---------- 3D Components ---------- */
function CameraDrift() {
  useFrame(({ camera, clock }) => {
    const t = clock.getElapsedTime();
    camera.position.x = Math.sin(t * 0.06) * 0.25;
    camera.position.y = 2.2 + Math.sin(t * 0.04) * 0.08;
    camera.lookAt(0, 0, 0);
  });
  return null;
}

function StarMesh({ size, paused }: { size: number; paused: boolean }) {
  const ref = useRef<THREE.Mesh>(null!);
  useFrame((_, dt) => {
    if (!paused && ref.current) ref.current.rotation.y += dt * 0.08;
  });
  return (
    <mesh ref={ref}>
      <sphereGeometry args={[size, 48, 48]} />
      <meshStandardMaterial
        color="#ffe9a8"
        emissive="#ffe9a8"
        emissiveIntensity={1.5}
        roughness={0.38}
      />
    </mesh>
  );
}

function Planet({
  label,
  orbitRadius,
  planetRadius,
  color,
  periodSec,
  paused,
  onHover,
  onClick,
  selected,
}: {
  label: string;
  orbitRadius: number;
  planetRadius: number;
  color: string;
  periodSec: number;
  paused: boolean;
  onHover: (h: boolean) => void;
  onClick: () => void;
  selected: boolean;
}) {
  const meshRef = useRef<THREE.Mesh>(null!);
  const theta = useRef(Math.random() * Math.PI * 2);

  useFrame((_, dt) => {
    const speed = (2 * Math.PI) / periodSec;
    theta.current += speed * dt * (paused ? 0.02 : 1);
    if (meshRef.current) {
      meshRef.current.position.set(
        orbitRadius * Math.cos(theta.current),
        0,
        orbitRadius * Math.sin(theta.current)
      );
    }
  });

  return (
    <group>
      {/* Orbit ring */}
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[orbitRadius - 0.001, orbitRadius + 0.001, 128]} />
        <meshBasicMaterial
          color={selected ? "#7dd3fc" : "#38bdf8"}
          transparent
          opacity={selected ? 0.85 : 0.35}
        />
      </mesh>

      {/* Planet */}
      <mesh
        ref={meshRef}
        onPointerOver={(e) => {
          e.stopPropagation();
          onHover(true);
        }}
        onPointerOut={(e) => {
          e.stopPropagation();
          onHover(false);
        }}
        onClick={(e) => {
          e.stopPropagation();
          onClick();
        }}
      >
        <sphereGeometry args={[planetRadius, 48, 48]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={0.7}
          roughness={0.4}
        />
        <Html center position={[0, planetRadius * 1.8, 0]}>
          <div
            style={{
              color: "white",
              background: "rgba(0,0,0,0.7)",
              padding: "4px 8px",
              borderRadius: "4px",
              fontSize: "12px",
              whiteSpace: "nowrap",
            }}
          >
            {label}
          </div>
        </Html>
      </mesh>
    </group>
  );
}

function OrbitSystem({
  detections,
  paused,
  selected,
  setSelected,
  setPaused,
}: {
  detections: Detection[];
  paused: boolean;
  selected: number | null;
  setSelected: (i: number) => void;
  setPaused: (p: boolean) => void;
}) {
  const planets = detections.map((d, i) => {
    const orbitRadius = 1.2 + i * 1.0;
    const planetRadius = 0.14 + (Math.min(Math.max(d.planet_radius_earth, 1), 6) / 10) * 0.25;
    const periodSec = 3 + Math.log2(Math.max(d.period_days, 0.4) + 1) * 2.2;
    const colors = ["#22d3ee", "#a78bfa", "#f472b6", "#34d399"];
    const color = colors[i % colors.length];
    return { label: d.name, orbitRadius, planetRadius, periodSec, color };
  });

  return (
    <group>
      {planets.map((p, i) => (
        <Planet
          key={i}
          label={p.label}
          orbitRadius={p.orbitRadius}
          planetRadius={p.planetRadius}
          color={p.color}
          periodSec={p.periodSec}
          paused={paused}
          onHover={(h) => setPaused(h)}
          onClick={() => setSelected(i)}
          selected={selected === i}
        />
      ))}
    </group>
  );
}

/* ---------- UI Components ---------- */
function LightCurve({ phase, flux }: { phase: number[]; flux: number[] }) {
  const sorted = [...flux].sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)] || 1;
  const norm = flux.map((f) => f / (median || 1));

  const absDevs = norm.map((v) => Math.abs(v - 1));
  const mad = [...absDevs].sort((a, b) => a - b)[Math.floor(absDevs.length / 2)] || 0.0002;
  const sigma = mad * 1.4826;

  const pad = Math.max(0.002, Math.min(0.02, 5 * sigma));
  const yMin = 1 - pad;
  const yMax = 1 + pad;

  const data = phase.map((p, i) => ({ phase: p, flux: norm[i] }));

  return (
    <div style={{ width: "100%", height: 180 }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 10, left: 0, bottom: 8 }}>
          <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
          <XAxis
            dataKey="phase"
            type="number"
            domain={[-0.5, 0.5]}
            tick={{ fill: "#a5b4fc" }}
            tickFormatter={(v) => (Math.abs(v) < 1e-6 ? "0" : v.toFixed(2))}
          />
          <YAxis
            domain={[yMin, yMax]}
            tick={{ fill: "#a5b4fc" }}
            tickFormatter={(v) => (v === 1 ? "1.000" : v.toFixed(3))}
          />
          <Tooltip
            contentStyle={{
              background: "#0f1220",
              border: "1px solid rgba(255,255,255,.08)",
              color: "#e2e8f0",
            }}
            formatter={(val: any) => [Number(val).toFixed(5), "Flux"]}
            labelFormatter={(l: any) => `Phase ${Number(l).toFixed(3)}`}
          />
          <ReferenceLine x={0} stroke="#60a5fa" strokeDasharray="4 4" />
          <ReferenceLine y={1} stroke="rgba(255,255,255,.25)" strokeDasharray="3 3" />
          <Line type="monotone" dataKey="flux" dot={false} stroke="#22d3ee" strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ---------- Main Visualizer Component ---------- */
export default function OrbitVisualizer({ starName, detections, onClose }: VisualizerProps) {
  const [paused, setPaused] = useState(false);
  const [selected, setSelected] = useState<number | null>(detections.length > 0 ? 0 : null);

  const selectedPlanet = selected !== null ? detections[selected] : null;

  return (
    <div style={{ position: "fixed", inset: 0, background: "#000", zIndex: 1000 }}>
      {/* Close button */}
      <button
        onClick={onClose}
        style={{
          position: "absolute",
          top: 16,
          right: 16,
          zIndex: 1001,
          background: "rgba(255,255,255,0.1)",
          border: "1px solid rgba(255,255,255,0.2)",
          color: "white",
          padding: "8px 16px",
          borderRadius: "8px",
          cursor: "pointer",
          fontSize: "14px",
        }}
      >
        Close Visualization
      </button>

      {/* Canvas */}
      <Canvas camera={{ position: [0, 2.2, 5.4], fov: 55 }}>
        <CameraDrift />
        <Stars radius={220} depth={100} count={9000} factor={4} saturation={0} fade speed={1} />
        <ambientLight intensity={0.55} />
        <pointLight position={[0, 0, 0]} intensity={2.0} color="#ffe9a8" />
        <StarMesh size={0.35} paused={paused} />
        {detections.length > 0 && (
          <OrbitSystem
            detections={detections}
            paused={paused}
            selected={selected}
            setSelected={setSelected}
            setPaused={setPaused}
          />
        )}
      </Canvas>

      {/* Planet info panel */}
      {selectedPlanet && (
        <motion.div
          initial={{ x: 360, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ type: "spring", stiffness: 120, damping: 14 }}
          style={{
            position: "absolute",
            right: 16,
            top: 76,
            width: 360,
            background: "rgba(15, 18, 32, 0.9)",
            backdropFilter: "blur(10px)",
            border: "1px solid rgba(255,255,255,0.1)",
            borderRadius: "12px",
            padding: "16px",
            color: "white",
          }}
        >
          <div style={{ fontWeight: "bold", marginBottom: 8, fontSize: "18px" }}>
            {starName} — {selectedPlanet.name}
          </div>
          <div style={{ marginBottom: 12 }}>
            <span
              style={{
                background: "rgba(34, 197, 94, 0.2)",
                color: "#22c55e",
                padding: "4px 8px",
                borderRadius: "4px",
                fontSize: "12px",
              }}
            >
              Confidence: {(selectedPlanet.confidence * 100).toFixed(1)}%
            </span>
          </div>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: "12px",
              marginBottom: 16,
            }}
          >
            <div style={{ background: "rgba(255,255,255,0.05)", padding: "12px", borderRadius: "8px" }}>
              <div style={{ fontSize: "12px", color: "#94a3b8", marginBottom: 4 }}>Period</div>
              <div style={{ fontWeight: "bold" }}>{selectedPlanet.period_days.toFixed(4)} d</div>
            </div>
            <div style={{ background: "rgba(255,255,255,0.05)", padding: "12px", borderRadius: "8px" }}>
              <div style={{ fontSize: "12px", color: "#94a3b8", marginBottom: 4 }}>Size vs Earth</div>
              <div style={{ fontWeight: "bold" }}>{selectedPlanet.planet_radius_earth.toFixed(2)} ×</div>
            </div>
            {selectedPlanet.transit_depth_ppm && (
              <div style={{ background: "rgba(255,255,255,0.05)", padding: "12px", borderRadius: "8px" }}>
                <div style={{ fontSize: "12px", color: "#94a3b8", marginBottom: 4 }}>Transit Depth</div>
                <div style={{ fontWeight: "bold" }}>{selectedPlanet.transit_depth_ppm.toFixed(0)} ppm</div>
              </div>
            )}
            {selectedPlanet.eq_temp_k && (
              <div style={{ background: "rgba(255,255,255,0.05)", padding: "12px", borderRadius: "8px" }}>
                <div style={{ fontSize: "12px", color: "#94a3b8", marginBottom: 4 }}>Temperature</div>
                <div style={{ fontWeight: "bold" }}>{Math.round(selectedPlanet.eq_temp_k)} K</div>
              </div>
            )}
          </div>

          {selectedPlanet.folded_lightcurve && (
            <LightCurve
              phase={selectedPlanet.folded_lightcurve.phase}
              flux={selectedPlanet.folded_lightcurve.flux}
            />
          )}
        </motion.div>
      )}

      {/* Star name label */}
      <div
        style={{
          position: "absolute",
          left: 16,
          top: 76,
          background: "rgba(15, 18, 32, 0.9)",
          backdropFilter: "blur(10px)",
          border: "1px solid rgba(255,255,255,0.1)",
          borderRadius: "12px",
          padding: "16px",
          color: "white",
        }}
      >
        <div style={{ fontWeight: "bold", fontSize: "18px", marginBottom: 4 }}>{starName}</div>
        <div style={{ fontSize: "14px", color: "#94a3b8" }}>
          {detections.length} planet{detections.length !== 1 ? "s" : ""} detected
        </div>
      </div>
    </div>
  );
}