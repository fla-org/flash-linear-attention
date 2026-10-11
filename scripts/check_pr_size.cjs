// Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/fla-org/flash-linear-attention/graphs/contributors

const ACKNOWLEDGEMENT = '- [x] I understand this PR exceeds 500 changed lines and have explained its single purpose ' +
  'and why it should stay together below.';

module.exports = function check_pr_size({ additions, deletions, body }) {
  if (![additions, deletions].every(value => Number.isInteger(value) && value >= 0)) {
    return 'Could not read the complete PR additions and deletions from GitHub.';
  }
  const changed_lines = additions + deletions;
  if (changed_lines <= 500) return null;

  let fence = null;
  const lines = (body || '').replace(/<!--[\s\S]*?(?:-->|$)/g, '').split(/\r?\n/).filter(line => {
    if (fence) {
      if (new RegExp(`^\\s*${fence[0]}{${fence.length},}\\s*$`).test(line)) fence = null;
      return false;
    }
    const opening = line.match(/^\s*(`{3,}|~{3,})/);
    if (opening) fence = opening[1];
    return !opening;
  });
  if (!lines.some(line => line.trim().replace(/^- \[X\]/, '- [x]') === ACKNOWLEDGEMENT)) {
    return `This PR changes ${changed_lines} lines (limit: 500). Tick the large-PR acknowledgement in the PR template.`;
  }

  const heading = lines.findIndex(line => line.trim() === '### Large PR justification');
  const section = [];
  if (heading >= 0) {
    for (let i = heading + 1; i < lines.length; i++) {
      const line = lines[i];
      if (/^\s*#{1,6}(?:\s|$)/.test(line) || /^\s*(?:=+|-+)\s*$/.test(lines[i + 1] || '')) break;
      section.push(line);
    }
  }
  const explained = section.join('\n').split(/\n\s*\n/).some(paragraph => {
    const text = paragraph.replace(/^\s*(?:[-*+]|\d+\.)\s+/gm, '').trim();
    const placeholder = /^(?:(?:todo|tbd|placeholder)\b|(?:n\/a|none)[.!]?$|<[^>]+>$|\[[^\]]+\]$)/i;
    return !placeholder.test(text) && text.replace(/\s/g, '').length >= 20;
  });
  if (!explained) {
    return 'Under "### Large PR justification", provide an explanation of the single purpose and why it should stay together ' +
      '(at least 20 non-whitespace characters). Comments, code fences, and placeholders do not count.';
  }
  return null;
};
