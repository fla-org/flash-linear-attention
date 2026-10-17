// Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/fla-org/flash-linear-attention/graphs/contributors

const assert = require('node:assert/strict');
const test = require('node:test');
const check_pr_commits = require('./check_pr_commits.cjs');

function commit(sha, message, parent_count = 1) {
  return {
    sha,
    commit: { message },
    parents: Array.from({ length: parent_count }, () => ({ sha: 'parent' })),
  };
}

test('tagged subjects pass, including multiline messages and excluded merge commits', () => {
  const commits = [
    commit('1111111', '[Fix] Guard checkpoint weight re-initialization'),
    commit('2222222', '[GDN2] Add bounded gate\n\nUse the existing tolerance.'),
    commit('3333333', 'Merge branch main into feature', 2),
  ];
  assert.equal(check_pr_commits(commits), null);
});

test('every invalid non-merge subject is reported', () => {
  const error = check_pr_commits([
    commit('aaaaaaa', 'fix: preserve packed boundaries'),
    commit('bbbbbbb', '[123] Use a numeric-only tag'),
    commit('ccccccc', '[Fix]'),
    commit('ddddddd', 'test: add regression'),
    commit('abababa', '[Fix]Missing space after bracket'),
    commit('eeeeeee', 'Merge branch main', 2),
  ]);
  assert.match(error, /single-commit squash/);
  assert.match(error, /aaaaaaa: fix: preserve packed boundaries/);
  assert.match(error, /bbbbbbb: \[123\] Use a numeric-only tag/);
  assert.match(error, /ccccccc: \[Fix\]/);
  assert.match(error, /ddddddd: test: add regression/);
  assert.match(error, /abababa: \[Fix\]Missing space after bracket/);
  assert.doesNotMatch(error, /eeeeeee/);
});

test('tags accept letters followed by letters or digits', () => {
  for (const message of ['[chore] Update CI', '[X] Support a new operator', '[NSA] Support cached inference']) {
    assert.equal(check_pr_commits([commit('fffffff', message)]), null);
  }
});
