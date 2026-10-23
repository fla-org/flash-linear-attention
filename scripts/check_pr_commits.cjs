// Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/fla-org/flash-linear-attention/graphs/contributors

const SUBJECT_PATTERN = /^\[[A-Za-z][A-Za-z0-9]*\]\s.+/;

module.exports = function check_pr_commits(commits) {
  const invalid = [];
  for (const { sha, commit, parents } of commits) {
    if (parents.length > 1) continue;
    const subject = commit.message.split(/\r?\n/, 1)[0];
    if (!SUBJECT_PATTERN.test(subject)) invalid.push(`${sha.slice(0, 7)}: ${subject}`);
  }
  if (invalid.length === 0) return null;
  return [
    'Every non-merge commit subject must start with a bracketed tag, for example "[Fix] Guard checkpoint weight re-initialization".',
    'A single-commit squash can use the commit subject as the final main title, so a valid PR title is not enough.',
    ...invalid,
  ].join('\n');
};
