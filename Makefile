USER = azureuser
IP = 20.71.92.52
REPO_ROOT = ~/recsys-competition
REMOTE = ${USER}@${IP}:${REPO_ROOT}
RSYNC = rsync \
		-e "ssh -i ${HOME}/.ssh/recsys-01.pem" \
		--archive \
		--human-readable \
		--partial \
		--recursive \
		--update \
		--info=PROGRESS2

upload:
	${RSYNC} --filter=':- .gitignore' . ${REMOTE}

download:
	${RSYNC} ${REMOTE}/results .
