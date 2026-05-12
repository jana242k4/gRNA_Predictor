#!/bin/bash
# One-time Oracle Cloud Ubuntu 22.04 ARM VM bootstrap.
# Run this once after SSH-ing into a fresh VM:
#   ssh ubuntu@<YOUR_ORACLE_VM_IP>
#   bash deploy.sh

set -e

echo "=== Step 1: System update ==="
sudo apt-get update && sudo apt-get upgrade -y

echo "=== Step 2: Install Docker (official apt repo, NOT snap) ==="
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Allow running docker without sudo
sudo usermod -aG docker $USER
echo "NOTE: Log out and back in for docker group to take effect, then re-run deploy.sh from Step 3."

echo "=== Step 3: Open firewall ports (Oracle iptables) ==="
# You ALSO need to open ports 80 and 443 in the Oracle Cloud Security List
# via the web console: Networking → Virtual Cloud Networks → Security Lists → Add Ingress Rules
sudo iptables -I INPUT 6 -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT 6 -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save

echo "=== Step 4: Clone repo ==="
cd ~
git clone https://github.com/jana242k4/gRNA_Predictor.git
cd gRNA_Predictor

echo "=== Step 5: Create .env file ==="
echo "Create the .env file with your secrets. At minimum:"
echo "  APP_ENV=production"
echo ""
echo "Run:  nano .env   (then paste your variables, Ctrl+X to save)"
read -p "Press Enter once you have created .env ..."

echo "=== Step 6: Build and start containers ==="
docker compose up -d --build

echo "=== Step 7: Verify ==="
sleep 10
curl -sf http://localhost/health && echo "Backend is up!" || echo "Check docker compose logs backend"

echo ""
echo "=== OPTIONAL: Free HTTPS with Let's Encrypt ==="
echo "If you have a domain pointing to this IP:"
echo "  sudo snap install --classic certbot"
echo "  sudo certbot certonly --standalone -d yourdomain.com"
echo "  Then uncomment the HTTPS server block in nginx.conf and run:"
echo "  docker compose restart nginx"
