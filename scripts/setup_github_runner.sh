#!/usr/bin/env bash
# Install a GitHub Actions self-hosted runner for this repository.
#
# Why: this repository is private, so every Actions minute is billed (docs/ci.md), and the port lanes
# are the expensive part of it -- they fetch multi-gigabyte checkpoints and run model inference. The
# machine that produced the benchmarks in CHANGELOG.md already has the three checkpoints and the three
# gfx1030 GPUs those lanes want, and a self-hosted runner spends no minutes. The workflows opt in
# through a repository variable, so nothing changes until it is set:
#
#     gh variable set LAYA_RUNNER --body '["self-hosted", "laya-gpu"]'
#
# Nothing here is needed for the blocking gate, which stays on hosted runners on purpose: a dead box
# must not be able to stop a push from being verified (docs/self-hosted-runner.md).
#
# Pinned. RUNNER_SHA256 is the linux-x64 checksum published in the v2.338.2 release notes, and it was
# re-derived from the downloaded tarball on 2026-11-06 with `sha256sum`; `--verify` re-checks it.
set -euo pipefail

RUNNER_VERSION="2.338.2"
RUNNER_SHA256="baebf204307bac6193480852732f0aba2ae97b6c5597c244f4d5558390147c40"
DEFAULT_REPO="thread3d/laya-words-sounds-vision"
DEFAULT_LABELS="laya-gpu"
DEFAULT_DIR="${HOME}/actions-runner"

usage() {
  cat <<'EOF'
Install a GitHub Actions self-hosted runner for this repository.

Usage:
  setup_github_runner.sh --token TOKEN [options]
  setup_github_runner.sh --verify TARBALL
  setup_github_runner.sh --dry-run [options]

Options:
  --token TOKEN     Registration token: Settings -> Actions -> Runners -> New runner.
                    Required unless --dry-run or --verify is used.
  --repo OWNER/REPO Target repository (default: thread3d/laya-words-sounds-vision).
  --labels LIST     Comma-separated runner labels (default: laya-gpu). The workflows ask for
                    ["self-hosted", "laya-gpu"], so changing this needs the variable changed too.
  --dir DIR         Install directory (default: ~/actions-runner).
  --tarball PATH    Use a tarball already on disk instead of downloading (offline installs).
  --dry-run         Print what would run, touch nothing.
  --verify TARBALL  Check a tarball against the pinned checksum and exit.
  --help            This text.

The service install needs sudo; the script prints that command rather than running it.
EOF
}

die() {
  echo "setup_github_runner: $*" >&2
  exit 1
}

require_linux_x64() {
  [[ "$(uname -s)" == "Linux" ]] || die "this script installs the linux-x64 runner; $(uname -s) is not supported"
  case "$(uname -m)" in
    x86_64 | amd64) ;;
    *) die "this script installs the linux-x64 runner; $(uname -m) is not supported" ;;
  esac
}

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

verify_tarball() {
  local path="$1" actual
  [[ -f "$path" ]] || die "no such tarball: $path"
  actual="$(sha256_of "$path")"
  if [[ "$actual" != "$RUNNER_SHA256" ]]; then
    die "checksum mismatch for $path
  want $RUNNER_SHA256
  got  $actual
The file is not the v${RUNNER_VERSION} linux-x64 runner this script pins; do not extract it."
  fi
  echo "setup_github_runner: $path matches the pinned v${RUNNER_VERSION} checksum"
}

main() {
  local repo="$DEFAULT_REPO" token="" labels="$DEFAULT_LABELS" dir="$DEFAULT_DIR"
  local tarball="" dry_run=0 verify=""

  while (($#)); do
    case "$1" in
      --token) token="${2:-}"; shift 2 || die "--token needs a value" ;;
      --repo) repo="${2:-}"; shift 2 || die "--repo needs a value" ;;
      --labels) labels="${2:-}"; shift 2 || die "--labels needs a value" ;;
      --dir) dir="${2:-}"; shift 2 || die "--dir needs a value" ;;
      --tarball) tarball="${2:-}"; shift 2 || die "--tarball needs a value" ;;
      --dry-run) dry_run=1; shift ;;
      --verify) verify="${2:-}"; shift 2 || die "--verify needs a tarball path" ;;
      --help | -h) usage; exit 0 ;;
      *) usage >&2; die "unknown argument: $1" ;;
    esac
  done

  if [[ -n "$verify" ]]; then
    verify_tarball "$verify"
    return 0
  fi

  require_linux_x64

  local url="https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"
  local file="${tarball:-/tmp/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz}"

  if ((dry_run)); then
    cat <<EOF
setup_github_runner (dry run -- nothing below is executed)
  runner          v${RUNNER_VERSION} (linux-x64), sha256 ${RUNNER_SHA256}
  repository      ${repo}
  labels          ${labels}
  install into    ${dir}
  download        ${tarball:+skipped, using ${tarball}}${tarball:-$url}
  verify          sha256 matches the pinned value
  configure       ./config.sh --unattended --url https://github.com/${repo} --token <token> \\
                    --labels ${labels} --replace --disableupdate
  service         sudo ./svc.sh install && sudo ./svc.sh start
  then enable     gh variable set LAYA_RUNNER --body '["self-hosted", "${labels%%,*}"]'
  roll back with  gh variable delete LAYA_RUNNER
EOF
    return 0
  fi

  [[ -n "$token" ]] || die "--token is required (Settings -> Actions -> Runners -> New runner); use --dry-run to see the plan"

  if [[ -z "$tarball" ]]; then
    command -v curl >/dev/null 2>&1 || die "curl is needed to download the runner; pass --tarball"
    echo "setup_github_runner: downloading $url"
    curl -sSfL -o "$file" "$url" || die "download failed; fetch $url yourself and pass --tarball"
  fi
  verify_tarball "$file"

  mkdir -p "$dir"
  tar xzf "$file" -C "$dir"
  [[ -x "$dir/config.sh" ]] || die "$dir/config.sh is missing after extraction"

  ( cd "$dir" && ./config.sh --unattended --url "https://github.com/${repo}" --token "$token" \
      --labels "$labels" --replace --disableupdate )

  cat <<EOF

setup_github_runner: configured. Two steps are left, both need sudo and are left to you:

  cd ${dir} && sudo ./svc.sh install && sudo ./svc.sh start

Then turn the lanes on and watch the runner pick one up:

  gh variable set LAYA_RUNNER --body '["self-hosted", "${labels%%,*}"]'
  gh api repos/${repo}/actions/runners --jq '.runners[] | "\\(.name) \\(.status) labels=\\(.labels|map(.name)|join(","))"'
  gh workflow run "Rust" --ref main

Roll back to hosted runners with: gh variable delete LAYA_RUNNER
EOF
}

main "$@"
