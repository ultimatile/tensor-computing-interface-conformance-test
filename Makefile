.PHONY: integration-check integration-test integration-restore

integration-check:
	./scripts/integration-check.sh check

integration-test:
	./scripts/integration-check.sh test

integration-restore:
	./scripts/integration-check.sh restore
