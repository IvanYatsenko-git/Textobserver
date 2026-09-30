import { Decoration, WidgetType } from '@codemirror/view';

const THESIS_COLORS = [
  '#FFE5B4',
  '#B4E5FF',
  '#D4FFB4',
  '#FFD4FF',
  '#FFFDB4',
  '#FFB4E5',
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
        attributes: { style: `background-color: ${color};` },
      })
    );
  }
  return colorMarkCache.get(idx);
}

const widgetCache = new Map();
function getDotsWidget(indices) {
  const key = indices.join(',');
  if (!widgetCache.has(key)) {
    widgetCache.set(key, new ThesisDotsWidget(indices));
  }
  return widgetCache.get(key);
}

function clampPos(pos, docLength) {
  return Math.max(0, Math.min(pos, docLength));
}

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
    docLength = 9999999,
  } = options;

  if (!segments || segments.length === 0) {
    return Decoration.none;
  }

  const thesisIndexMap = new Map();
  theses.forEach((t, i) => thesisIndexMap.set(t.id, i));

  const ranges = [];
  const hasActiveTheses = activeTheses.length > 0;

  for (const segment of segments) {
    const from = clampPos(segment.startChar, docLength);
    const to = clampPos(segment.endChar, docLength);

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
          } else if (
            typeof status.score === 'number' &&
            status.score >= threshold
          ) {
            relevant.push({
              idx: thesisIndexMap.get(thesisId),
              score: status.score,
            });
          }
        }

        if (relevant.length > 0) {
          relevant.sort((a, b) => b.score - a.score || a.idx - b.idx);
          const primaryIdx = relevant[0].idx;
          ranges.push(getColorMark(primaryIdx).range(from, to));

          if (relevant.length > 1) {
            const extraIndices = relevant.slice(1).map((r) => r.idx);
            ranges.push(
              Decoration.widget({
                widget: getDotsWidget(extraIndices),
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

  ranges.sort((a, b) => a.from - b.from || a.to - b.to);
  return Decoration.set(ranges, true);
}

export function getThesisColor(idx) {
  return THESIS_COLORS[idx % THESIS_COLORS.length];
}



