# Video Object Cropping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a browser-based video preview app where a user uploads a video, clicks a person once, and sees an adaptive crop that follows that person.

**Architecture:** A small Vite + React + TypeScript app will render the source video in a canvas, run browser-side person detection with TensorFlow.js COCO-SSD, and use pure tracking/crop helpers to keep the selected person centered in a second preview canvas. Detection runs on the main thread for the MVP, with the pure math isolated so it can be tested independently and moved behind a worker later if needed.

**Tech Stack:** React, Vite, TypeScript, TensorFlow.js, `@tensorflow-models/coco-ssd`, Vitest, Testing Library.

---

### Task 1: Scaffold the app and core geometry helpers

**Files:**
- Create: `package.json`
- Create: `index.html`
- Create: `tsconfig.json`
- Create: `tsconfig.node.json`
- Create: `vite.config.ts`
- Create: `src/main.tsx`
- Create: `src/App.tsx`
- Create: `src/styles.css`
- Create: `src/lib/geometry.ts`
- Create: `src/lib/geometry.test.ts`

- [ ] **Step 1: Write the failing test**

```ts
import { describe, expect, it } from 'vitest';
import { computeCropRect } from './geometry';

describe('computeCropRect', () => {
  it('centers a crop box around the tracked box and clamps to frame bounds', () => {
    const result = computeCropRect(
      { x: 300, y: 200, width: 120, height: 240 },
      { width: 1280, height: 720 },
      9 / 16,
    );

    expect(result.width / result.height).toBeCloseTo(9 / 16, 2);
    expect(result.x).toBeGreaterThanOrEqual(0);
    expect(result.y).toBeGreaterThanOrEqual(0);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/lib/geometry.test.ts`
Expected: FAIL because `computeCropRect` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```ts
export type Box = { x: number; y: number; width: number; height: number };
export type FrameSize = { width: number; height: number };

export function computeCropRect(box: Box, frame: FrameSize, aspectRatio: number): Box {
  return box;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/lib/geometry.test.ts`
Expected: PASS after the real crop math is implemented.

- [ ] **Step 5: Commit**

```bash
git add package.json index.html tsconfig.json tsconfig.node.json vite.config.ts src/main.tsx src/App.tsx src/styles.css src/lib/geometry.ts src/lib/geometry.test.ts
git commit -m "feat: scaffold crop preview geometry"
```

### Task 2: Add selection and tracking helpers

**Files:**
- Create: `src/lib/tracking.ts`
- Create: `src/lib/tracking.test.ts`
- Modify: `src/App.tsx`

- [ ] **Step 1: Write the failing test**

```ts
import { describe, expect, it } from 'vitest';
import { choosePersonForClick } from './tracking';

describe('choosePersonForClick', () => {
  it('returns the person box containing the click point', () => {
    const selection = choosePersonForClick(
      [
        { label: 'person', score: 0.93, box: { x: 10, y: 10, width: 80, height: 100 } },
        { label: 'person', score: 0.88, box: { x: 150, y: 20, width: 90, height: 110 } },
      ],
      { x: 35, y: 40 },
    );

    expect(selection?.box.x).toBe(10);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/lib/tracking.test.ts`
Expected: FAIL because the selection helper does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```ts
export type Box = { x: number; y: number; width: number; height: number };
export type Detection = { label: string; score: number; box: Box };
export type Point = { x: number; y: number };

export function choosePersonForClick(detections: Detection[], point: Point): Detection | null {
  return detections.find((d) =>
    point.x >= d.box.x &&
    point.x <= d.box.x + d.box.width &&
    point.y >= d.box.y &&
    point.y <= d.box.y + d.box.height,
  ) ?? null;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/lib/tracking.test.ts`
Expected: PASS once the helper is complete.

- [ ] **Step 5: Commit**

```bash
git add src/lib/tracking.ts src/lib/tracking.test.ts src/App.tsx
git commit -m "feat: add click target selection helpers"
```

### Task 3: Build the upload, detection, and preview UI

**Files:**
- Modify: `src/App.tsx`
- Modify: `src/styles.css`
- Create: `src/lib/model.ts`

- [ ] **Step 1: Write the failing integration expectation**

```ts
it('shows a preview prompt after a video is loaded', () => {
  render(<App />);
  expect(screen.getByText(/upload a video/i)).toBeInTheDocument();
});
```

- [ ] **Step 2: Run the test to verify the current UI does not satisfy the preview state**

Run: `npx vitest run src/App.test.tsx`
Expected: FAIL until the app renders the upload flow and preview state.

- [ ] **Step 3: Implement the UI**

```tsx
export default function App() {
  return (
    <main>
      <section>
        <h1>Video Crop Preview</h1>
        <input type="file" accept="video/*" />
      </section>
    </main>
  );
}
```

- [ ] **Step 4: Run the app and verify browser behavior**

Run: `npm run dev`
Expected: App loads, accepts video upload, renders source canvas, allows target selection, and shows the adaptive crop preview.

- [ ] **Step 5: Commit**

```bash
git add src/App.tsx src/styles.css src/lib/model.ts
git commit -m "feat: build video crop preview workflow"
```

### Task 4: Polish and verify

**Files:**
- Modify: any files needed from Tasks 1-3

- [ ] **Step 1: Run the full test suite**

Run: `npm test`
Expected: All tests pass.

- [ ] **Step 2: Run the production build**

Run: `npm run build`
Expected: Successful build with no TypeScript or Vite errors.

- [ ] **Step 3: Validate the app in the browser**

Run: `npm run dev`
Expected: A user can upload a video, click a person, and watch the preview crop follow them with a clear reacquiring state if tracking drifts.

