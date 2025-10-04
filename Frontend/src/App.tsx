import React, { useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Stars, Html } from "@react-three/drei";
import * as THREE from "three";
import { motion, AnimatePresence } from "framer-motion";
import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, ReferenceLine,
} from "recharts";

/* ---------- Types ---------- */
type StarMeta = { name?: string; ra?: number; dec?: number; teff?: number; mass_sun?: number; radius_sun?: number; };
type Detection = {
  name: string; prob: number; period_days: number; a_rs: number; rp_re: number;
  depth?: number; eq_temp_k?: number; folded?: { phase: number[]; flux: number[] }; flags?: [string, string][];
};
type AppState = "landing" | "searching" | "results";

/* ---------- Demo fallback so UI works before backend ---------- */
function makeFold() {
  const phase = Array.from({ length: 220 }, (_, i) => -0.5 + i / 220);
  const flux = phase.map(p => 1 + Math.exp(-((p / 0.03) ** 2)) * -0.0012 + (Math.random() - 0.5) * 0.00018);
  return { phase, flux };
}
function demo(query: string) {
  const star: StarMeta = { name: query, ra: 285.679, dec: 50.241, teff: 5700, mass_sun: 0.91, radius_sun: 1.07 };
  const detections: Detection[] = [
    { name: `${query}-b`, prob: 0.88, period_days: 0.84, a_rs: 3.2, rp_re: 1.6, depth: 0.0012, eq_temp_k: 1700, folded: makeFold(), flags: [["Odd/Even match","✓"],["No secondary @0.5","✓"]] },
    { name: `${query}-c`, prob: 0.72, period_days: 5.2, a_rs: 8.0, rp_re: 2.2, depth: 0.0018, eq_temp_k: 900, folded: makeFold(), flags: [["Odd/Even match","✓"]] },
  ];
  return { star, detections };
}

/** Fetch from your backend. It should:
 * - return {notFound: true} if the star doesn’t exist
 * - else return { star: {...}, detections: [...] }
 */
async function fetchAnalyze(target: string): Promise<{ notFound?: boolean; star?: StarMeta; detections: Detection[] }> {
  try {
    const r = await fetch(`/api/analyze?target=${encodeURIComponent(target)}`);
    if (!r.ok) throw new Error(await r.text());
    const j = await r.json();
    if (j?.notFound) return { notFound: true, detections: [] };
    const star: StarMeta = j.star ?? { name: j.id };
    const detections: Detection[] = (j.detections || []).map((d: any, i: number) => ({
      name: d.name || `${j.id}-p${i+1}`,
      prob: d.prob ?? 0.8,
      period_days: d.period_days ?? 3,
      a_rs: d.a_rs ?? (3 + i*2),
      rp_re: d.rp_re ?? (1.5 + i*0.7),
      depth: d.features?.depth,
      eq_temp_k: d.eq_temp_k,
      folded: d.folded,
      flags: d.flags,
    }));
    return { star, detections };
  } catch {
    // remove this demo() when your backend is ready
    return demo(target);
  }
}

/* ---------- 3D helpers ---------- */
function CameraDrift() {
  useFrame(({ camera, clock }) => {
    const t = clock.getElapsedTime();
    const r = 0.25;
    camera.position.x = Math.sin(t * 0.06) * r;
    camera.position.y = 2.2 + Math.sin(t * 0.04) * 0.08;
    camera.lookAt(0, 0, 0);
  });
  return null;
}

function StarMesh({ size, paused, onClick }: { size: number; paused: boolean; onClick: () => void }) {
  const ref = useRef<THREE.Mesh>(null!);
  useFrame((_, dt) => { if (!paused) ref.current.rotation.y += dt * 0.08; });
  return (
    <mesh ref={ref} onClick={(e)=>{e.stopPropagation(); onClick();}}>
      <sphereGeometry args={[size, 48, 48]} />
      <meshStandardMaterial color={"#ffe9a8"} emissive={"#ffe9a8"} emissiveIntensity={1.5} roughness={0.38}/>
    </mesh>
  );
}


function Planet({
  label, a, r, color, periodSec, paused, onHover, onClick, selected,
}: { label: string; a: number; r: number; color: string; periodSec: number; paused: boolean; onHover: (h:boolean)=>void; onClick:()=>void; selected:boolean; }) {
  const meshRef = useRef<THREE.Mesh>(null!);
  const theta = useRef(Math.random() * Math.PI * 2);
  useFrame((_, dt) => {
    const speed = (2 * Math.PI) / periodSec;
    theta.current += speed * dt * (paused ? 0.02 : 1);
    meshRef.current.position.set(a * Math.cos(theta.current), 0, a * Math.sin(theta.current));
  });
  return (
    <group>
      <mesh rotation={[-Math.PI/2, 0, 0]}>
        <ringGeometry args={[a - 0.001, a + 0.001, 128]} />
        <meshBasicMaterial color={selected ? "#7dd3fc" : "#38bdf8"} transparent opacity={selected ? 0.85 : 0.35}/>
      </mesh>
      <mesh
        ref={meshRef}
        onPointerOver={(e)=>{e.stopPropagation(); onHover(true);}}
        onPointerOut={(e)=>{e.stopPropagation(); onHover(false);}}
        onClick={(e)=>{e.stopPropagation(); onClick();}}
      >
        <sphereGeometry args={[r, 48, 48]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.7} roughness={0.4}/>
        <Html center position={[0, r*1.8, 0]}>
          <div className="tooltip">{label}</div>
        </Html>
      </mesh>
    </group>
  );
}

function OrbitSystem({
  detections, paused, selected, setSelected, setPaused,
}: { detections: Detection[]; paused: boolean; selected: number | null; setSelected:(i:number)=>void; setPaused:(p:boolean)=>void; }) {
  const planets = useMemo(() => detections.map((d, i) => {
    const orbitRadius = 1.2 + i * 1.0 + (d.a_rs - 3) * 0.05;
    const planetRadius = 0.14 + (Math.min(Math.max(d.rp_re, 1), 6) / 10) * 0.25;
    const periodSec = 3 + Math.log2(Math.max(d.period_days, 0.4) + 1) * 2.2;
    const color = ["#22d3ee", "#a78bfa", "#f472b6", "#34d399"][i % 4];
    return { label: d.name, orbitRadius, planetRadius, periodSec, color };
  }), [detections]);
  return (
    <group>
      {planets.map((p, i) => (
        <Planet
          key={i} label={p.label} a={p.orbitRadius} r={p.planetRadius} color={p.color}
          periodSec={p.periodSec} paused={paused}
          onHover={(h)=>setPaused(h)} onClick={()=>setSelected(i)} selected={selected===i}
        />
      ))}
    </group>
  );
}

/* ---------- UI helpers ---------- */
function LightCurve({ phase, flux }: { phase: number[]; flux: number[] }) {
  // --- 1) Robust normalization around 1.0 ---
  const sorted = [...flux].sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)] || 1;
  const norm = flux.map(f => f / (median || 1));

  // Robust spread via MAD (median absolute deviation)
  const absDevs = norm.map(v => Math.abs(v - 1));
  const mad = [...absDevs].sort((a, b) => a - b)[Math.floor(absDevs.length / 2)] || 0.0002;
  const sigma = mad * 1.4826; // MAD -> sigma

  // Autoscale: at least ±0.2% (0.002) so tiny dips are visible, but don’t over-zoom.
  const pad = Math.max(0.002, Math.min(0.02, 5 * sigma)); // between 0.2% and 2%
  const yMin = 1 - pad;
  const yMax = 1 + pad;

  // Build data for recharts
  const data = phase.map((p, i) => ({ phase: p, flux: norm[i] }));

  return (
    <div style={{ width: "100%", height: 180 }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 10, left: 0, bottom: 8 }}>
          <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false}/>
          <XAxis
            dataKey="phase"
            type="number"
            domain={[-0.5, 0.5]}
            tick={{ fill: "#a5b4fc" }}
            tickFormatter={(v)=> (Math.abs(v) < 1e-6 ? "0" : v.toFixed(2))}
          />
          <YAxis
            domain={[yMin, yMax]}
            tick={{ fill: "#a5b4fc" }}
            tickFormatter={(v)=> (v === 1 ? "1.000" : v.toFixed(3))}
          />
          <Tooltip
            contentStyle={{ background: "#0f1220", border: "1px solid rgba(255,255,255,.08)", color: "#e2e8f0" }}
            formatter={(val:any, name:any)=> [Number(val).toFixed(5), name === "flux" ? "Flux" : name]}
            labelFormatter={(l:any)=> `Phase ${Number(l).toFixed(3)}`}
          />
          {/* Reference lines */}
          <ReferenceLine x={0} stroke="#60a5fa" strokeDasharray="4 4" />
          <ReferenceLine y={1} stroke="rgba(255,255,255,.25)" strokeDasharray="3 3" />
          <Line type="monotone" dataKey="flux" dot={false} stroke="#22d3ee" strokeWidth={2}/>
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="card">
      <div className="muted">{label}</div>
      <div className="bold">{value}</div>
    </div>
  );
}

/* ---------- Landing hero (centered search) ---------- */
function LandingHero({
  query, setQuery, onSearch,
}: { query: string; setQuery: (v:string)=>void; onSearch: ()=>void; }) {
  const chips = ["Kepler-10", "KIC 11446443", "TIC 25155310", "19:02:43.2 +50:14:21"];
  return (
    <div className="center">
      <motion.div
        className="search-card"
        initial={{ y: 10, opacity: 0 }} animate={{ y: 0, opacity: 1 }}
        transition={{ type: "spring", stiffness: 120, damping: 12 }}
      >
        <div className="title">Find new worlds.</div>
        <div className="subtitle">Discover exoplanets using NASA data and AI.</div>

        <div className="row">
          <input
            autoFocus className="input big"
            value={query}
            placeholder="Enter star name, catalog ID, or coordinates (RA, Dec)…"
            onChange={(e)=>setQuery(e.target.value)}
            onKeyDown={(e)=> e.key === "Enter" && onSearch()}
          />
          <button className="btn cta" onClick={onSearch}>Search</button>
        </div>

        <div className="chips">
          {chips.map((c)=>(
            <button key={c} className="chip" onClick={()=>setQuery(c)}>{c}</button>
          ))}
        </div>
      </motion.div>
    </div>
  );
}

function RadarScan({
  color = "#49ffc8",
  inner = 0.55,   // just outside the star (your star radius ~0.35)
  outer = 3.0,    // how far the sweep extends visually
  speed = 0.9,    // radians/second
}: { color?: string; inner?: number; outer?: number; speed?: number }) {
  // Rotating sweep
  const sweepRef = useRef<THREE.Mesh>(null!);

  // Three expanding pulse rings
  const rings = [
    useRef<THREE.Mesh>(null!),
    useRef<THREE.Mesh>(null!),
    useRef<THREE.Mesh>(null!),
  ] as const;
  const mats = [
    useRef<THREE.MeshBasicMaterial>(null!),
    useRef<THREE.MeshBasicMaterial>(null!),
    useRef<THREE.MeshBasicMaterial>(null!),
  ] as const;

  // Thin sector of a ring (about 22.5 degrees)
  const thetaLength = Math.PI / 8;

  useFrame((_, dt) => {
    // rotate sweep around Z (group is rotated so plane is XZ)
    if (sweepRef.current) sweepRef.current.rotation.z -= speed * dt;

    // expanding, fading pings
    const t = performance.now() / 1000;
    rings.forEach((r, i) => {
      const phase = (t + i * 0.5) % 1;                 // 0 → 1 loop, staggered
      const scale = 1 + phase * (outer / inner - 1);   // grow from inner → outer
      r.current.scale.setScalar(scale);
      mats[i].current.opacity = 0.35 * (1 - phase);    // fade out
    });
  });

  return (
    <group rotation={[-Math.PI / 2, 0, 0]}>
      {/* pings */}
      {[0, 1, 2].map((i) => (
        <mesh key={i} ref={rings[i]}>
          <ringGeometry args={[inner, inner + 0.006, 128]} />
          <meshBasicMaterial ref={mats[i]} color={color} transparent opacity={0.35} />
        </mesh>
      ))}

      {/* faint guide ring */}
      <mesh>
        <ringGeometry args={[inner, inner + 0.001, 256]} />
        <meshBasicMaterial color={color} transparent opacity={0.18} />
      </mesh>

      {/* rotating sweep sector */}
      <mesh ref={sweepRef}>
        {/* ring sector: inner→outer, 256 segments, thetaStart=0, thetaLength */}
        <ringGeometry args={[inner, outer, 256, 1, 0, thetaLength]} />
        <meshBasicMaterial color={color} transparent opacity={0.22} side={THREE.DoubleSide} />
      </mesh>
    </group>
  );
}


/* ---------- Main App ---------- */
export default function App() {
  const [state, setState] = useState<AppState>("landing");
  const [query, setQuery] = useState("Kepler-10");
  const [paused, setPaused] = useState(false);
  const [star, setStar] = useState<StarMeta | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [selected, setSelected] = useState<number | null>(0);
  const [showStarPanel, setShowStarPanel] = useState(false);

  async function onSearch() {
    if (!query.trim()) return;
    // Move to "searching" and show star+radar only if the target resolves.
    setState("searching");
const started = performance.now();
const res = await fetchAnalyze(query.trim());

    if (res.notFound) {
      // back to landing (centered search), no star shown
      setStar(null);
      setDetections([]);
      setSelected(null);
      setState("landing");
      return;
    }

    const minMs = 600;
const elapsed = performance.now() - started;
const goResults = () => setState("results");
elapsed < minMs ? setTimeout(goResults, minMs - elapsed) : goResults();

    setStar(res.star ?? { name: query.trim() });
    setDetections(res.detections);
    setSelected(res.detections.length ? 0 : null);
    setShowStarPanel(false);

    // If detections exist, go to results; otherwise still show results view with none.
    setState("results");
  }

  /* Sticky top bar appears only after a valid search */
  const TopBar = state === "landing" ? null : (
    <div className="topbar">
      <div className="brand">🔭 Exoplanet AI</div>
      <div className="search-wrap">
        <input
          className="cosmic-input"
          placeholder="Star name, catalog ID, or coordinates (RA, Dec)…"
          value={query}
          onChange={(e)=>setQuery(e.target.value)}
          onKeyDown={(e)=> e.key === "Enter" && onSearch()}
        />
        <button className="cosmic-btn" onClick={onSearch}>Search</button>
      </div>
      <div className="topbar-spacer" />
    </div>
  );

  /* Right planet panel */
  const PlanetPanel = () => {
    if (state !== "results" || selected == null || !detections.length) return null;
    const d = detections[selected];
    return (
      <motion.div
        initial={{ x: 360, opacity: 0 }} animate={{ x: 0, opacity: 1 }}
        transition={{ type: "spring", stiffness: 120, damping: 14 }}
        style={{ position: "absolute", right: 16, top: 76, width: 360, zIndex: 10 }}
        className="card"
      >
        <div className="bold" style={{ marginBottom: 6 }}>{(star?.name ?? query)} — {d.name}</div>
        <div className="row" style={{ gap: 8, marginBottom: 8 }}>
          <span className="badge ok">P(planet) {(d.prob*100).toFixed(1)}%</span>
          {(d.flags || []).map(([msg, b], i)=>(
            <span key={i} className={`badge ${b==='✗'?'no':b==='⚠️'?'warn':'ok'}`}>{b} {msg}</span>
          ))}
        </div>
        <div className="grid2">
          <StatCard label="Period" value={`${d.period_days.toFixed(4)} d`} />
          <StatCard label="Size vs Earth" value={`${d.rp_re.toFixed(2)} ×`} />
          <StatCard label="a / R★" value={d.a_rs.toFixed(2)} />
          <StatCard label="T_eq" value={`${Math.round(d.eq_temp_k ?? 1000)} K`} />
        </div>
        {d.folded && <LightCurve phase={d.folded.phase} flux={d.folded.flux!} />}
      </motion.div>
    );
  };

  /* Left star panel (click the star to toggle) */
  const StarPanel = () => {
    if (!star || state === "landing" || !showStarPanel) return null;
    return (
      <motion.div
        initial={{ x: -360, opacity: 0 }} animate={{ x: 0, opacity: 1 }}
        transition={{ type: "spring", stiffness: 120, damping: 14 }}
        style={{ position: "absolute", left: 16, top: 76, width: 320, zIndex: 10 }}
        className="card"
      >
        <div className="bold" style={{ marginBottom: 6 }}>{star.name ?? query}</div>
        <div className="grid2">
          <StatCard label="RA" value={star.ra ? `${star.ra.toFixed(3)}°` : "—"} />
          <StatCard label="Dec" value={star.dec ? `${star.dec.toFixed(3)}°` : "—"} />
          <StatCard label="T_eff" value={star.teff ? `${Math.round(star.teff)} K` : "—"} />
          <StatCard label="Mass" value={star.mass_sun ? `${star.mass_sun.toFixed(2)} M☉` : "—"} />
          <StatCard label="Radius" value={star.radius_sun ? `${star.radius_sun.toFixed(2)} R☉` : "—"} />
        </div>
        <div className="muted" style={{ marginTop: 10, fontSize: 12 }}>Tip: Click the star to hide/show this panel.</div>
      </motion.div>
    );
  };

  return (
    <div style={{ height: "100%", position: "relative" }}>
      {TopBar}

      <Canvas camera={{ position: [0, 2.2, 5.4], fov: 55 }}>
        <CameraDrift />
        <Stars radius={220} depth={100} count={9000} factor={4} saturation={0} fade speed={1} />
        <ambientLight intensity={0.55} />
        <pointLight position={[0,0,0]} intensity={2.0} color={"#ffe9a8"} />

        {/* Show star + radar ONLY after a valid search (searching/results) */}
        {state !== "landing" && (
  <>
    <StarMesh size={0.35} paused={paused} onClick={() => setShowStarPanel(s => !s)} />
    {state === "searching" && <RadarScan />}   {/* rotating sweep + pings */}
  </>
)}


        {/* Planets only in results */}
        {state === "results" && detections.length > 0 && (
          <OrbitSystem
            detections={detections}
            paused={paused}
            selected={selected}
            setSelected={setSelected}
            setPaused={setPaused}
          />
        )}
      </Canvas>

      {/* Animated nebula overlay */}
      <div className="overlay-grad animated-nebula" />

      {/* Landing center search ONLY when nothing has been searched or star not found */}
      {state === "landing" && (
        <LandingHero query={query} setQuery={setQuery} onSearch={onSearch} />
      )}

      {/* Panels */}
      <StarPanel />
      <PlanetPanel />
    </div>
  );
}
