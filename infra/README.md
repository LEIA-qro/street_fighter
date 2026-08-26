# Infra de 'la madre' — coordinador ES en AWS

'La madre' es la VPS que coordina el entrenamiento por Evolution Strategies: reparte
seeds a los workers (nuestras maquinas), agrega los fitness que regresan, actualiza la
politica y sube checkpoints a S3. Los workers viven en las laptops/desktop del equipo y
hablan con ella por Tailscale; **nada de esta infra escucha en internet**.

Es infraestructura **desechable**: se levanta en ~3 minutos, se tira con un comando y no
deja nada en la cuenta.

---

## ⚠️ Cuenta correcta

Esto va en la cuenta de **EDUCACION** (perfil `awsedu`), **nunca** en produccion.

Si nunca has usado AWS desde esta maquina, primero instala el CLI y configura el
perfil SSO (una sola vez):

```bash
# macOS: brew install awscli | Windows: winget install Amazon.AWSCLI | WSL: sudo apt install awscli
aws configure sso   # con el start URL / credenciales de la cuenta de educacion
                    # (hoy ese acceso lo tiene Felipe); profile name: awsedu; region: us-west-2
```

Antes de cualquier `terraform`, asegurate (y re-corre `aws sso login --profile awsedu`
cuando el token expire, ~8-12 h):

```bash
# bash / zsh / WSL
export AWS_PROFILE=awsedu
aws sts get-caller-identity   # el Account debe ser el de educacion
```

```powershell
# PowerShell (Windows)
$env:AWS_PROFILE = "awsedu"
aws sts get-caller-identity
```

**Costo:** una t3.small son ~15 USD/mes (~0.02 USD/hora) + ~2 USD/mes del disco de 20GB
+ centavos de S3. Y es apagable: `terraform destroy` en cualquier momento y el costo se
detiene. No hay pretexto para dejarla prendida un fin de semana sin entrenar.

## 1. Instalar Terraform

```bash
# macOS
brew tap hashicorp/tap && brew install hashicorp/tap/terraform

# Windows (PowerShell como admin)
choco install terraform     # o: winget install HashiCorp.Terraform

# Ubuntu/WSL
wget -O - https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
sudo apt update && sudo apt install terraform
```

## 2. Conseguir el auth key de Tailscale

1. Entra a <https://login.tailscale.com/admin/settings/keys> (la tailnet del equipo).
2. **Generate auth key** con:
   - **Reusable** ✅ — por si la instancia se recrea (p. ej. al cambiar user_data).
   - **Ephemeral** ✅ — el nodo se borra solo de la tailnet cuando la instancia muere;
     congruente con infra desechable: `destroy` tampoco deja basura en Tailscale.
3. Copia el `tskey-auth-...` al tfvars (siguiente paso). Expira en 90 dias max; si la
   madre vive mas que eso, genera otro.

> **Nota de exposicion (aplica a los TRES secretos — tailscale key, W&B key y el token
> de GitHub si usas esa via):** viajan dentro del user_data, que cualquier proceso local
> de la VPS puede leer via IMDS y cualquier admin de la cuenta via
> `ec2:DescribeInstanceAttribute`. Por eso los tres son de bajo privilegio: el tailscale
> key es ephemeral/reusable (revocable en el admin panel), el token de GitHub es
> solo-lectura de un repo, y la W&B key solo escribe metricas. Si alguno se quema:
> revocar y `terraform apply` de nuevo.

## 3. Clonado del repo en la VPS (elige UNA)

El default de `repo_url` es la URL SSH (`git@github.com:LEIA-qro/street_fighter.git`) y
una VPS recien nacida **no tiene con que autenticarse** a GitHub. Dos opciones:

- **Token https (la facil, recomendada):** en GitHub → Settings → Developer settings →
  Fine-grained tokens, crea un token con acceso de **solo lectura a Contents** del repo
  `LEIA-qro/street_fighter`, y pasa:
  `repo_url = "https://oauth2:github_pat_XXXX@github.com/LEIA-qro/street_fighter.git"`.
  El token queda en el user_data (visible para admins de la cuenta AWS) — por eso
  solo-lectura y de un solo repo.
- **Deploy key (la pro):** genera un par `ssh-keygen -t ed25519`, sube la publica en
  GitHub → repo → Settings → Deploy keys (sin write). La privada hay que ponerla en
  `/root/.ssh/id_ed25519` de la VPS **antes** de que corra el clone, cosa que el
  user_data no hace por ti — tendrias que entrar con `tailscale ssh`, ponerla y correr
  el clone a mano. Util si el repo se vuelve privado-paranoico; para el dia a dia usa
  el token.

## 4. Levantar

```bash
cd infra
cp terraform.tfvars.example terraform.tfvars
# edita terraform.tfvars: tailscale_auth_key, owner, (wandb_api_key, repo_url...)

terraform init
terraform plan     # revisa: 1 instancia, 1 SG, 1 bucket, 1 rol — nada mas
terraform apply    # escribe 'yes'
```

O sin tfvars, todo por flags:

```bash
terraform apply -var tailscale_auth_key=tskey-auth-... -var owner=tu-usuario
```

El primer boot tarda ~2-3 min (instala tailscale, clona, arma el venv, arranca el
servicio). El progreso vive en `/var/log/leia-user-data.log` dentro de la instancia.

**Estado local a proposito:** no hay backend remoto. El `terraform.tfstate` se queda en
tu maquina (y esta git-ignorado porque contiene los secretos). Esta bien asi: la infra
es desechable, la levanta una sola persona a la vez, y un backend S3/DynamoDB seria
justo el residuo permanente que queremos que no exista. Consecuencia: **quien hace
`apply` hace `destroy`** — el tfstate no se comparte.

## 5. Verificar

```bash
# En la consola AWS (region us-west-2): EC2 → Instances → filtra por tag
#   Project = leia-sf2-es
# Todo lo de este proyecto (instancia, volumen, SG, bucket, rol) trae ese tag.

tailscale status | grep madre          # el nodo aparece en tu tailnet
tailscale ssh madre                    # shell en la VPS, sin llaves ni puerto 22

# ya adentro:
journalctl -u leia-coordinator -f      # logs en vivo del coordinador
systemctl status leia-coordinator

# desde cualquier maquina de la tailnet:
curl http://madre:8080/status
```

## 6. Metricas

En la madre **no** hay TensorBoard: las metricas del coordinador van a **Weights &
Biases** (si pasaste `wandb_api_key`). Miralas en
`https://wandb.ai/<tu-entidad>/leia-sf2-es` (o el `wandb_project` que hayas puesto).
Los checkpoints quedan en `/opt/leia/checkpoints` y, con `create_bucket=true`, en el
bucket S3 que reporta `terraform output checkpoint_bucket`.

## 7. Destruir (y dejar la cuenta limpia)

```bash
terraform destroy   # escribe 'yes'
```

Se lleva instancia, volumen, security group, rol/perfil IAM y el bucket **con todo y
checkpoints** (`force_destroy=true` — si algun checkpoint importa, bajalo antes con
`aws s3 sync s3://$(terraform output -raw checkpoint_bucket) ./ckpt-backup`). Como el
nodo Tailscale es ephemeral, tambien desaparece solo de la tailnet. Despues del destroy
la cuenta queda exactamente como antes; verificalo si quieres con el filtro
`Project=leia-sf2-es` en la consola: cero resultados.
