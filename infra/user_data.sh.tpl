#!/usr/bin/env bash
# user_data de 'la madre'. Corre UNA vez, como root, en el primer boot (cloud-init).
# Log completo en /var/log/leia-user-data.log — es lo primero que hay que leer si el
# coordinador no aparece. Ojo template: si algun dia se agrega un $ de bash, se escapa
# como $${var} (sintaxis de templatefile() de Terraform); hoy el archivo no usa ninguno.
#
# SIN -x a proposito: xtrace imprimiria cada comando expandido — incluidos el auth key
# de Tailscale y un token de GitHub si repo_url lo trae — a este log (0644) y a
# /var/log/cloud-init-output.log. Los echo de progreso de abajo lo sustituyen.
set -euo pipefail
exec > >(tee -a /var/log/leia-user-data.log) 2>&1

export DEBIAN_FRONTEND=noninteractive

# --- Tailscale: unica via de acceso (el SG no tiene ingress; todo entra por el tunel) ---
echo "[user-data] instalando tailscale..."
curl -fsSL https://tailscale.com/install.sh | sh

# --ssh habilita Tailscale SSH: no hay llaves que administrar ni puerto 22 expuesto.
# --hostname=madre fija el nombre con el que la ve la flota (curl http://madre:8080).
# Si el auth key esta atado a un tag de ACL, agregar aqui --advertise-tags=tag:madre.
tailscale up \
  --auth-key='${tailscale_auth_key}' \
  --hostname=madre \
  --ssh

# --- Dependencias del coordinador -------------------------------------------------------
apt-get update -y
apt-get install -y python3-venv python3-pip git

# --- Codigo -----------------------------------------------------------------------------
# accept-new: primer contacto con github.com sin prompt interactivo (no hay TTY aqui).
# Con la URL SSH default esto requiere una deploy key ya instalada; la alternativa sin
# llaves es repo_url https con token (ver infra/README.md).
echo "[user-data] clonando repo (rama ${repo_branch})..."
export GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=accept-new"
git clone --branch '${repo_branch}' --single-branch '${repo_url}' /opt/leia

python3 -m venv /opt/leia/.venv
/opt/leia/.venv/bin/pip install --upgrade pip
# El coordinador solo necesita numpy (agrega vectores de fitness); wandb para metricas,
# boto3 para subir checkpoints. Sin torch: la madre no evalua politicas.
/opt/leia/.venv/bin/pip install numpy wandb boto3
# Si el repo trae el pin exacto del track ES, respeta ese por encima del install suelto.
if [ -f /opt/leia/requirements-es.txt ]; then
  /opt/leia/.venv/bin/pip install -r /opt/leia/requirements-es.txt
fi

mkdir -p /opt/leia/checkpoints

# --- Servicio ---------------------------------------------------------------------------
# Restart=always: el coordinador debe sobrevivir a sus propios crashes y a reboots de la
# instancia; los workers reintentan la conexion solos, asi que reiniciarlo es gratis.
cat > /etc/systemd/system/leia-coordinator.service <<'UNIT'
[Unit]
Description=LEIA SF2 ES coordinator (la madre)
After=network-online.target tailscaled.service
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/opt/leia
${wandb_env_line}
ExecStart=/opt/leia/.venv/bin/python src/es/coordinator.py --host 0.0.0.0 --port 8080 --checkpoint-dir /opt/leia/checkpoints${s3_flag} --wandb-project ${wandb_project}
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

# El unit lleva WANDB_API_KEY en Environment=; systemd lo lee como root, nadie
# mas necesita leerlo. (El user-data crudo sigue visible via IMDS para procesos
# locales -- documentado en infra/README.md.)
chmod 600 /etc/systemd/system/leia-coordinator.service

echo "[user-data] habilitando servicio leia-coordinator..."
systemctl daemon-reload
systemctl enable --now leia-coordinator.service
echo "[user-data] listo."
