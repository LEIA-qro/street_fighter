# Infraestructura — cómo operar (2026-08-26)

## La madre (coordinador ES)
- EC2 t3.small us-east-1, cuenta AWS **educación** (perfil `awsedu`, 800407728644 — NUNCA la de producción). Creada con terraform (infra/), TODO etiquetado Project=leia-sf2-es. `terraform destroy` la tira sin rastro (bucket force_destroy; borrar el nodo "madre" del console de Tailscale a mano después).
- Acceso: `tailscale ssh ubuntu@madre` (SG sin ingress; todo por Tailscale). Repo en /opt/leia (owned root → usar sudo para git).
- Servicio: `leia-coordinator` (unit 0600 por el WANDB_API_KEY adentro). Operaciones típicas:
  - actualizar: `sudo git -C /opt/leia pull && sudo systemctl restart leia-coordinator`
  - logs: `sudo journalctl -u leia-coordinator -f`
  - flags actuales del ExecStart: `--host 0.0.0.0 --port 8080 --checkpoint-dir /opt/leia/checkpoints --s3-bucket leia-sf2-es-ckpt-d4a2b8dc99bbd0b480e7520f5e --wandb-project leia-sf2-es --states manifest --difficulty 1`
- Estado del run es identidad: para RUN NUEVO limpiar /opt/leia/checkpoints y `aws s3 rm s3://leia-sf2-es-ckpt-.../es/ --recursive` (AWS_PROFILE=awsedu); si no, resume del último checkpoint (S3 restore incluido).
- AWS SSO expira ~8-12h: `aws sso login --sso-session focaltec` (abre browser de Felipe). us-west-2 NO tiene VPC default en esta cuenta; terraform usa us-east-1.

## Tailscale
- Tailnet de la ORG GitHub: `leia-qro.org.github` (los 3 owners = admins). Plan gratis OK: la madre es nodo NORMAL, NO ephemeral (el plan gratis mide nodos ephemeral: 1,000 min/mes — una madre 24/7 los quema en un día). Key expiry de la madre: deshabilitado.
- Gotcha del console: el formulario de auth keys RESETEA los toggles al corregir campos — verificar Reusable/Ephemeral ANTES de Generate. Los keys usados fueron single-use por eso.
- Nodos: madre (100.90.13.19), mini-fzamorano (M4). Faltan: omen(-wsl), legion(-wsl), desktop(-wsl).

## W&B
- Team `leia-qro-rl`, proyecto `leia-sf2-es` (privado). Admins: @felipaupz, @santiago64; Diego invitado (diegop00dx@gmail.com, pendiente de aceptar). Solo la MADRE necesita el API key (va en terraform.tfvars local gitignoreado y en el unit); los workers nunca lo tocan. El run del coordinador aparece "running" siempre (es el latido del servicio, no un entrenamiento).
- PPO/BizHawk NO reporta a W&B (TensorBoard local en la desktop). Mejora pendiente: sync SB3→W&B.

## Secretos — dónde viven (nunca en git)
- infra/terraform.tfvars (gitignoreado, en la M4): tailscale auth key + wandb api key.
- El user_data los lleva a la instancia (visibles vía IMDS para admins de la cuenta — por diseño, los 3 son de bajo privilegio y revocables).

## Máquinas
- desktop "SSS" (i9-13900K/4090, Windows, D:\GitHub\Street\street_fighter): rig BizHawk + PPO. Pendiente su WSL2 para ser worker.
- Omen (275HX/5080M) y Legion (275HX/5070TiM): pendientes de alta completa (tools/setup_worker.md).
- M4 (mini-fzamorano, 24GB): worker retro + máquina de Felipe. Stub ~/TEC/LEIA/EmuHawk.exe permite importar core.config aquí.
