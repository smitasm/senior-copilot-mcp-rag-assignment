.PHONY: test run-alarm run-ticketing run-gui up down logs build

# Note: `make` isn't native to Windows. These are conveniences for
# Linux/Mac contributors and CI; every target's underlying command is also
# documented directly in README.md so a Windows/PowerShell user is never
# blocked on not having make installed.

test:
	python -m pytest tests -q

run-alarm:
	python -m uvicorn alarm_api.main:app --port 8000

run-ticketing:
	python -m uvicorn ticketing_api.main:app --port 8001

run-gui:
	python -m streamlit run gui/app.py

build:
	docker compose build

up:
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f
