// Block-based segmentation with accurate position tracking.
//
// Algorithm:
//   1. Split the original text into "blocks" — chunks separated by blank lines.
//      Each block carries its exact (startChar, endChar) in the original text.
//   2. Group blocks into segments:
//        - A header always starts a new segment.
//        - Consecutive headers form a header chain (e.g. H1 followed by H2).
//        - The first paragraph after a header chain joins that segment.
//        - Subsequent paragraphs accumulate until the segment reaches
//          MIN_SEGMENT_LENGTH; the next paragraph then starts a new segment.
//   3. Post-pass: any segment shorter than SHORT_SEGMENT_THRESHOLD merges
//      backward into the previous segment (handles trailing fragments,
//      image references, captions, "keywords" lines, etc.).

const SENTENCE_REGEX = /[.!?]+(?=\s|$)/g;
const MIN_SEGMENT_LENGTH = 220;
const SHORT_SEGMENT_THRESHOLD = 100;

function isAtxHeader(line) {
  const trimmed = line.trim();
  return trimmed.length > 0 && /^#{1,6}\s+/.test(trimmed);
}

function isSetextUnderline(line) {
  const trimmed = line.trim();
  if (!trimmed) return false;
  return /^=+$/.test(trimmed) || /^-{2,}$/.test(trimmed);
}

function blockIsHeader(blockText) {
  const lines = blockText.split('\n');
  for (const line of lines) {
    if (isAtxHeader(line)) return true;
    if (isSetextUnderline(line)) return true;
  }
  return false;
}

function countSentences(text) {
  return (text.match(SENTENCE_REGEX) || []).length;
}

// Walk the raw text, emitting one block per non-empty paragraph.
// Tracks startChar (start of content) and endChar (one past the last
// non-newline character), so positions stay accurate regardless of
// whitespace between blocks.
function parseBlocks(text) {
  const blocks = [];
  const len = text.length;
  let pos = 0;

  while (pos < len) {
    while (pos < len && text[pos] === '\n') pos++;
    if (pos >= len) break;

    const startChar = pos;
    let lastContentEnd = pos;

    while (pos < len) {
      if (text[pos] === '\n') {
        if (pos + 1 < len && text[pos + 1] === '\n') break;
      } else {
        lastContentEnd = pos + 1;
      }
      pos++;
    }

    const endChar = lastContentEnd;
    const blockText = text.substring(startChar, endChar);

    if (blockText.trim()) {
      blocks.push({
        text: blockText,
        startChar,
        endChar,
        isHeader: blockIsHeader(blockText),
      });
    }
  }

  return blocks;
}

function groupBlocks(blocks) {
  const segments = [];
  let current = null;

  const startSegment = (block) => {
    current = {
      startChar: block.startChar,
      endChar: block.endChar,
      blocks: [block],
    };
  };

  const addBlock = (block) => {
    current.blocks.push(block);
    current.endChar = block.endChar;
  };

  const finalize = () => {
    if (current) {
      segments.push(current);
      current = null;
    }
  };

  const currentLength = () =>
    current ? current.endChar - current.startChar : 0;

  const currentIsHeaderChainOnly = () =>
    current && current.blocks.every((b) => b.isHeader);

  for (const block of blocks) {
    if (!current) {
      startSegment(block);
      continue;
    }

    if (block.isHeader) {
      if (currentIsHeaderChainOnly()) {
        addBlock(block);
      } else {
        finalize();
        startSegment(block);
      }
    } else {
      if (currentIsHeaderChainOnly()) {
        addBlock(block);
      } else if (currentLength() < MIN_SEGMENT_LENGTH) {
        addBlock(block);
      } else {
        finalize();
        startSegment(block);
      }
    }
  }
  finalize();

  // Merge any remaining short segments backward into the previous one.
  for (let i = segments.length - 1; i > 0; i--) {
    const segLen = segments[i].endChar - segments[i].startChar;
    if (segLen < SHORT_SEGMENT_THRESHOLD) {
      const prev = segments[i - 1];
      prev.endChar = segments[i].endChar;
      prev.blocks = prev.blocks.concat(segments[i].blocks);
      segments.splice(i, 1);
    }
  }

  // Edge case: the very first segment is short and there's another to merge into.
  if (
    segments.length > 1 &&
    segments[0].endChar - segments[0].startChar < SHORT_SEGMENT_THRESHOLD
  ) {
    const merged = {
      startChar: segments[0].startChar,
      endChar: segments[1].endChar,
      blocks: segments[0].blocks.concat(segments[1].blocks),
    };
    segments.splice(0, 2, merged);
  }

  return segments;
}

export function segmentText(text) {
  if (!text || !text.trim()) return [];

  const blocks = parseBlocks(text);
  const grouped = groupBlocks(blocks);

  return grouped.map((seg, idx) => {
    const segmentText = text.substring(seg.startChar, seg.endChar);
    return {
      id: `seg_${idx}`,
      text: segmentText,
      startChar: seg.startChar,
      endChar: seg.endChar,
      sentenceCount: countSentences(segmentText),
    };
  });
}

export function getAllSegmentIds(segments) {
  return segments.map((s) => s.id);
}

export function getUnknownSegments(segments, coverage) {
  return segments.filter((segment) => {
    const segmentCoverage = coverage[segment.id];
    if (!segmentCoverage) return true;
    return Object.values(segmentCoverage).some(
      (status) => status.status === 'unknown'
    );
  });
}

export function hasSignificantChange(oldText, newText) {
  if (!oldText) return true;
  const oldLength = oldText.length;
  const newLength = newText.length;
  const change = Math.abs(oldLength - newLength) / oldLength;
  return change > 0.25;
}
