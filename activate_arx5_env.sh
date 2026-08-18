# Source this file before running ARX5 / diffusion-policy commands:
#   source ./activate_arx5_env.sh

ARX5_PROJECT_ROOT="${ARX5_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}"
ARX5_SDK_ROOT="$ARX5_PROJECT_ROOT/arx5-sdk-main"
ARX5_DP_ROOT="$ARX5_PROJECT_ROOT/diffusion_policy-main"
ARX5_CONDA_SH="${ARX5_CONDA_SH:-${HOME}/mambaforge/etc/profile.d/conda.sh}"
ARX5_CONDA_ENV="${ARX5_CONDA_ENV:-arx5-dp}"

if [ -f "$ARX5_CONDA_SH" ]; then
  # shellcheck disable=SC1090
  . "$ARX5_CONDA_SH"
  conda activate "$ARX5_CONDA_ENV"
else
  echo "WARNING: conda setup script not found: $ARX5_CONDA_SH" >&2
  echo "Please activate conda env manually: conda activate $ARX5_CONDA_ENV" >&2
fi

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$ARX5_SDK_ROOT/lib/x86_64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ARX5_SDK_ROOT/python:$ARX5_DP_ROOT:${PYTHONPATH:-}"

arx5_fix_proxy_var() {
  var_name="$1"
  eval "var_value=\${$var_name:-}"
  if [ -n "$var_value" ] && \
     [ "${var_value#http://}" = "$var_value" ] && \
     [ "${var_value#https://}" = "$var_value" ]; then
    export "$var_name=http://$var_value"
  fi
}

for proxy_var in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY; do
  arx5_fix_proxy_var "$proxy_var"
done

cd "$ARX5_DP_ROOT" || return 1

echo "ARX5 environment ready."
echo "  conda env: $ARX5_CONDA_ENV"
echo "  cwd: $PWD"
echo "  SDK lib: $ARX5_SDK_ROOT/lib/x86_64"
