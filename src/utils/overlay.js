import { Decoration, WidgetType } from '@codemirror/view';

const THESIS_COLORS = [
  '#FFE5B4', // peach
  '#B4E5FF', // light blue
  '#D4FFB4', // light green
  '#FFD4FF', // light magenta
  '#FFFDB4', // light yellow
  '#FFB4E5', // light pink
];

class PilcrowWidget extends WidgetType {
  toDOM() {
    const span = document.createElement('span');
    span.className = 'cm-pilcrow';
    span.textContent = '¶';
    return span;
  }

  eq(other) {
    return other instanceof PilcrowWidget;
  }
}

class ThesisDotsWidget extends WidgetType {
  constructor(colorIndices) {
    super();
    this.colorIndices = colorIndices;
  }

  toDOM() {
    const span = document.createElement('span');
    span.className = 'cm-thesis-dots';
    for (const idx of this.colorIndices) {
      const dot = document.createElement('span');
      dot.className = 'cm-thesis-dot';
      dot.style.backgroundColor = THESIS_COLORS[idx % THESIS_COLORS.length];
      span.appendChild(dot);
    }
    return span;
  }

  eq(other) {
    return (
      other instanceof ThesisDotsWidget &&
      this.colorIndices.length === other.colorIndices.length &&
      this.colorIndices.every((c, i) => c === other.colorIndices[i])
    );
  }
}

const pilcrowWidget = new PilcrowWidget();
const pilcrowDeco = Decoration.widget({ widget: pilcrowWidget, side: 2 });

const colorMarkCache = new Map();
const unknownMark = Decoration.mark({ class: 'coverage-unknown' });
const segmentBoundaryLine = Decoration.line({ class: 'cm-segment-boundary' });

function getColorMark(idx) {
  if (!colorMarkCache.has(idx)) {
    const color = THESIS_COLORS[idx % THESIS_COLORS.length];
    colorMarkCache.set(
      idx,
      Decoration.mark({
        class: 'coverage-covered',
        // ВИПРАВЛЕНО: Додано лапки навколо значення властивості style
        attributes: { style: `background-color: ${color};` },
      })
    );
  }
  return colorMarkCache.get(idx);
}

// ─── ДОДАНО: Кеш для віджетів ─────────────────────────────────────────
const widgetCache = new Map();
function getDotsWidget(indices) {
  const key = indices.join(',');
  if (!widgetCache.has(key)) {
    widgetCache.set(key, new ThesisDotsWidget(indices));
  }
  return widgetCache.get(key);
}

// ─── ДОДАНО: Helper для обрізання координат (clamping) ────────────────
function clampPos(pos, docLength) {
  return Math.max(0, Math.min(pos, docLength));
}

// Build all segment-level decorations in one pass:
//   - background colors per coverage status
//   - line decorations for segment boundaries (debug)
//   - pilcrow widgets at segment ends
//
// Returns a sorted Decoration set ready to dispatch.
export function buildSegmentDecorations(
  segments,
  coverage,
  activeTheses,
  theses,
  options = {}
) {
  const {
    showBoundaries = false,
    showPilcrows = false,
    threshold = 60,
    docLength = 9999999 // Додано: довжина документа для clamping (якщо забудуть передати)
  } = options;

  if (!segments || segments.length === 0) {
    return Decoration.none;
  }

  const thesisIndexMap = new Map();
  theses.forEach((t, i) => thesisIndexMap.set(t.id, i));

  const ranges = [];
  const hasActiveTheses = activeTheses.length > 0;

  for (const segment of segments) {
    // ─── ВАЖЛИВО: Отримуємо безпечні координати для CM6 ───
    const from = clampPos(segment.startChar, docLength);
    const to = clampPos(segment.endChar, docLength);

    // Якщо після clamping координати зламались, пропускаємо цей сегмент
    if (from >= to) continue;

    if (showBoundaries) {
      ranges.push(segmentBoundaryLine.range(from));
    }

    if (hasActiveTheses) {
      const segCov = coverage[segment.id];
      if (segCov) {
        let hasUnknown = false;
        const relevant = [];

        for (const thesisId of activeTheses) {
          const status = segCov[thesisId];
          if (!status) continue;
          if (status.status === 'unknown') {
            hasUnknown = true;
          } else if (status.status === 'covered') {
            const numericScore = Number(status.score);
            if (Number.isFinite(numericScore) && numericScore >= threshold) {
              relevant.push({
                idx: thesisIndexMap.get(thesisId),
                score: numericScore,
              });
            }
          }
        }

        if (relevant.length > 0) {
          relevant.sort((a, b) => b.score - a.score || a.idx - b.idx);
          const primaryIdx = relevant[0].idx;
          
          // Використовуємо from/to замість segment.startChar/endChar
          ranges.push(getColorMark(primaryIdx).range(from, to));

          if (relevant.length > 1) {
            const extraIndices = relevant.slice(1).map((r) => r.idx);
            ranges.push(
              Decoration.widget({
                widget: getDotsWidget(extraIndices), // Використовуємо кеш
                side: 1,
              }).range(to)
            );
          }
        } else if (hasUnknown) {
          ranges.push(unknownMark.range(from, to));
        }
      }
    }

    if (showPilcrows) {
      ranges.push(pilcrowDeco.range(to));
    }
  }

  // ─── ВАЖЛИВО: CodeMirror 6 ВИМАГАЄ, щоб декорації були відсортовані ───
  ranges.sort((a, b) => a.from - b.from || a.to - b.to);

  return Decoration.set(ranges, true);
}

export function getThesisColor(idx) {
  return THESIS_COLORS[idx % THESIS_COLORS.length];
}