// Track coverage state for segments: covered, unknown, uncovered
// When text is edited, mark adjacent segments as unknown

export const COVERAGE_STATUS = {
  COVERED: 'covered',
  UNKNOWN: 'unknown',
  UNCOVERED: 'uncovered',
};

// Initialize coverage for a segment with all theses
export function initializeSegmentCoverage(segmentId, theses) {
  const coverage = {};

  for (const thesis of theses) {
    coverage[thesis.id] = {
      status: COVERAGE_STATUS.UNKNOWN,
      score: 0,
    };
  }

  return coverage;
}

// Mark segments as unknown due to text edit
export function markAdjacentSegmentsUnknown(editStart, editEnd, segments, coverage) {
  const unknownSegmentIds = [];

  for (const segment of segments) {
    // If edit overlaps with segment boundary or is adjacent, mark as unknown
    if (
      (editStart > segment.startChar && editStart < segment.endChar) ||
      (editEnd > segment.startChar && editEnd < segment.endChar) ||
      (editStart <= segment.startChar && editEnd >= segment.startChar)
    ) {
      unknownSegmentIds.push(segment.id);
    }

    // Mark adjacent segments as unknown
    const BUFFER = 50; // Char buffer for adjacency
    if (
      Math.abs(editEnd - segment.startChar) < BUFFER ||
      Math.abs(editStart - segment.endChar) < BUFFER
    ) {
      unknownSegmentIds.push(segment.id);
    }
  }

  return unknownSegmentIds;
}

// Update segment coverage with Gemini classification results
export function updateCoverage(segmentId, thesisId, classification) {
  return {
    status: classification.relevant
      ? COVERAGE_STATUS.COVERED
      : COVERAGE_STATUS.UNCOVERED,
    score: classification.score || 0,
  };
}

// Get segments that need Gemini classification
export function getSegmentsNeedingClassification(segments, coverage, theses) {
  const needsClassification = [];

  for (const segment of segments) {
    const segmentCoverage = coverage[segment.id] || {};

    for (const thesis of theses) {
      const status = segmentCoverage[thesis.id];

      if (!status || status.status === COVERAGE_STATUS.UNKNOWN) {
        needsClassification.push({
          segmentId: segment.id,
          thesisId: thesis.id,
          segmentText: segment.text,
          thesisText: thesis.title,
        });
      }
    }
  }

  return needsClassification;
}
