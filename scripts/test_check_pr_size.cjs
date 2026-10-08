// Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/fla-org/flash-linear-attention/graphs/contributors

const assert = require('node:assert/strict');
const test = require('node:test');
const check_pr_size = require('./check_pr_size.cjs');

const ACK = '- [x] I understand this PR exceeds 500 changed lines and have explained its single purpose ' +
  'and why it should stay together below.';
const HEADING = '### Large PR justification';
const REASON = 'This adds one operator; its implementation, reference, and tests must land together to verify its contract.';
const BODY = `${ACK}\n\n${HEADING}\n\n${REASON}`;

test('500 changed lines pass; 501 require acknowledgement, counting additions and deletions', () => {
  for (const [additions, deletions] of [[500, 0], [0, 500], [250, 250]]) {
    assert.equal(check_pr_size({ additions, deletions, body: null }), null);
  }
  for (const [additions, deletions] of [[501, 0], [0, 501], [250, 251]]) {
    assert.match(check_pr_size({ additions, deletions, body: null }), /acknowledgement/);
  }
});

test('acknowledgement must be checked, exact, and outside comments and code fences', () => {
  for (const body of [BODY.replace(ACK, ''), BODY.replace('[x]', '[ ]'), BODY.replace('500', '600'),
    `<!--\n${ACK}\n-->\n${HEADING}\n${REASON}`, `\`\`\`md\n${ACK}\n\`\`\`\n${HEADING}\n${REASON}`]) {
    assert.match(check_pr_size({ additions: 501, deletions: 0, body }), /acknowledgement/);
  }
});

test('rationale cannot be empty, placeholder text, a comment, code, or an adjacent section', () => {
  for (const reason of ['', 'Too short.', '<Explain the single purpose and why it should stay together>',
    '[Describe the reason these changes need to stay together]', 'TODO: explain why this PR must remain together',
    `<!--\n${REASON}\n-->`, `\`\`\`text\n${REASON}\n\`\`\``, `~~~text\n${REASON}\n~~~`,
    `\n## Test plan\n${REASON}`, `\n### Another section\n${REASON}`, '- TODO: fill in the large PR reason',
    `The following section is not a justification\n===\n${REASON}`]) {
    const body = `${ACK}\n\n${HEADING}\n\n${reason}`;
    assert.match(check_pr_size({ additions: 501, deletions: 0, body }), /explanation/);
  }
  assert.match(check_pr_size({ additions: 501, deletions: 0, body: `${ACK}\n${REASON}` }), /explanation/);
});

test('valid rationale passes and ignores non-content around the paragraph', () => {
  for (const body of [BODY, BODY.replaceAll('\n', '\r\n'), BODY.replace('[x]', '[X]'), BODY.replace(REASON, `- ${REASON}`),
    `${ACK}\n${HEADING}\n<!-- instructions -->\n\n${REASON}\n\n## Tests\nNot yet run.`]) {
    assert.equal(check_pr_size({ additions: 600, deletions: 200, body }), null);
  }
});

test('updated PR totals and body are evaluated on each run', () => {
  const pr = { additions: 490, deletions: 10, body: null };
  assert.equal(check_pr_size(pr), null);
  pr.additions++;
  assert.match(check_pr_size(pr), /acknowledgement/);
  pr.body = BODY;
  assert.equal(check_pr_size(pr), null);
  pr.body = BODY.replace(REASON, '');
  assert.match(check_pr_size(pr), /explanation/);
  pr.additions--;
  assert.equal(check_pr_size(pr), null);
});

test('missing API statistics fail rather than bypassing the limit', () => {
  assert.match(check_pr_size({ body: BODY }), /Could not read/);
});
