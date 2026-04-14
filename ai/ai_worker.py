# -*- coding: utf-8 -*-
"""
ai_worker.py
------------
QThread che esegue le chiamate API in background, senza bloccare l'interfaccia.

Provider supportati (rilevati automaticamente dal nome del modello):
  - Anthropic  → modello contiene "claude"
  - Google     → modello contiene "gemini"
  - OpenAI-compatible → tutto il resto (gpt-*, deepseek-*, mistral-*, ...)
"""

import json
import requests
from PyQt5 import QtCore


class AIWorker(QtCore.QThread):
    """Thread worker per le chiamate API AI."""

    # Segnali
    risposta_ricevuta = QtCore.pyqtSignal(str)   # Testo risposta completo
    errore_ricevuto   = QtCore.pyqtSignal(str)   # Messaggio errore
    stato_aggiornato  = QtCore.pyqtSignal(str)   # Messaggi di stato

    TIMEOUT = 90  # secondi

    def __init__(self, modello: str, api_key: str, messaggi: list, system_prompt: str, parent=None):
        super().__init__(parent)
        self.modello       = modello.strip()
        self.api_key       = api_key.strip()
        self.messaggi      = messaggi        # lista di {"role": "user"|"assistant", "content": str}
        self.system_prompt = system_prompt

    # ------------------------------------------------------------------
    # ENTRY POINT DEL THREAD
    # ------------------------------------------------------------------

    def run(self):
        if not self.api_key:
            self.errore_ricevuto.emit("⚠ Chiave API non inserita.")
            return
        if not self.modello:
            self.errore_ricevuto.emit("⚠ Modello non specificato.")
            return

        try:
            self.stato_aggiornato.emit("Connessione in corso...")
            modello_lower = self.modello.lower()

            if "claude" in modello_lower:
                risposta = self._chiama_anthropic()
            elif "gemini" in modello_lower:
                risposta = self._chiama_gemini()
            elif "deepseek" in modello_lower:
                risposta = self._chiama_deepseek()
            else:
                risposta = self._chiama_openai()

            self.risposta_ricevuta.emit(risposta)

        except requests.exceptions.Timeout:
            self.errore_ricevuto.emit("⚠ Timeout: il server non ha risposto in tempo.")
        except requests.exceptions.ConnectionError:
            self.errore_ricevuto.emit("⚠ Errore di connessione. Verifica la connessione internet.")
        except requests.exceptions.HTTPError as e:
            self._gestisci_http_error(e)
        except Exception as e:
            self.errore_ricevuto.emit(f"⚠ Errore inaspettato: {str(e)}")

    # ------------------------------------------------------------------
    # ANTHROPIC API  (claude-*)
    # ------------------------------------------------------------------

    def _chiama_anthropic(self) -> str:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "Content-Type":      "application/json",
            "x-api-key":         self.api_key,
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model":      self.modello,
            "max_tokens": 4096,
            "system":     self.system_prompt,
            "messages":   self.messaggi,
        }

        self.stato_aggiornato.emit("Richiesta inviata ad Anthropic...")
        resp = requests.post(url, headers=headers, json=payload, timeout=self.TIMEOUT)
        resp.raise_for_status()

        data = resp.json()
        testo = ""
        for blocco in data.get("content", []):
            if blocco.get("type") == "text":
                testo += blocco.get("text", "")
        return testo

    # ------------------------------------------------------------------
    # GOOGLE GEMINI API  (gemini-*)
    # ------------------------------------------------------------------

    def _chiama_gemini(self) -> str:
        """
        Chiama l'API Google Gemini (v1beta generateContent).
        Il system prompt viene passato come campo 'system_instruction'.
        La chiave API va nell'URL come query parameter.
        """
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.modello}:generateContent?key={self.api_key}"
        )
        headers = {"Content-Type": "application/json"}

        # Converti la storia messaggi nel formato Gemini
        # Gemini usa "user" e "model" (non "assistant")
        contents = []
        for msg in self.messaggi:
            ruolo = "model" if msg["role"] == "assistant" else "user"
            contents.append({
                "role": ruolo,
                "parts": [{"text": msg["content"]}]
            })

        payload = {
            "system_instruction": {
                "parts": [{"text": self.system_prompt}]
            },
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": 4096,
                "temperature": 0.7,
            }
        }

        self.stato_aggiornato.emit("Richiesta inviata a Google Gemini...")
        resp = requests.post(url, headers=headers, json=payload, timeout=self.TIMEOUT)
        resp.raise_for_status()

        data = resp.json()

        # Estrae il testo dalla risposta Gemini
        try:
            candidates = data.get("candidates", [])
            if not candidates:
                return "(Nessuna risposta ricevuta da Gemini)"
            parts = candidates[0].get("content", {}).get("parts", [])
            return "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError):
            return "(Formato risposta Gemini non riconosciuto)"

    # ------------------------------------------------------------------
    # DEEPSEEK API  (deepseek-chat, deepseek-reasoner, ...)
    # ------------------------------------------------------------------

    def _chiama_deepseek(self) -> str:
        """
        Chiama l'API DeepSeek (api.deepseek.com).
        Formato identico a OpenAI, ma endpoint e modelli propri.
        Modelli principali: deepseek-chat, deepseek-reasoner
        """
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        messaggi_completi = [
            {"role": "system", "content": self.system_prompt}
        ] + self.messaggi

        payload = {
            "model":      self.modello,
            "messages":   messaggi_completi,
            "max_tokens": 4096,
        }

        self.stato_aggiornato.emit("Richiesta inviata a DeepSeek...")
        resp = requests.post(url, headers=headers, json=payload, timeout=self.TIMEOUT)
        resp.raise_for_status()

        data = resp.json()
        scelte = data.get("choices", [])
        if not scelte:
            return "(Nessuna risposta ricevuta da DeepSeek)"
        return scelte[0].get("message", {}).get("content", "")

    # ------------------------------------------------------------------
    # OPENAI-COMPATIBLE API  (gpt-*, mistral-*, ...)
    # ------------------------------------------------------------------

    def _chiama_openai(self) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        messaggi_completi = [
            {"role": "system", "content": self.system_prompt}
        ] + self.messaggi

        payload = {
            "model":      self.modello,
            "messages":   messaggi_completi,
            "max_tokens": 4096,
        }

        self.stato_aggiornato.emit("Richiesta inviata al provider...")
        resp = requests.post(url, headers=headers, json=payload, timeout=self.TIMEOUT)
        resp.raise_for_status()

        data = resp.json()
        scelte = data.get("choices", [])
        if not scelte:
            return "(Nessuna risposta ricevuta)"
        return scelte[0].get("message", {}).get("content", "")

    # ------------------------------------------------------------------
    # GESTIONE ERRORI HTTP
    # ------------------------------------------------------------------

    def _gestisci_http_error(self, e: requests.exceptions.HTTPError):
        codice = e.response.status_code if e.response is not None else "?"
        try:
            corpo = e.response.json()
            # Compatibile con Anthropic, OpenAI e Gemini (campi diversi)
            err = corpo.get("error", {})
            dettaglio = (
                err.get("message")          # Anthropic / OpenAI
                or err.get("status")        # Gemini
                or str(e)
            )
        except Exception:
            dettaglio = str(e)

        if codice == 400:
            self.errore_ricevuto.emit(f"⚠ Richiesta non valida (400): {dettaglio}")
        elif codice == 401:
            self.errore_ricevuto.emit("⚠ Chiave API non valida (401). Verifica la chiave inserita.")
        elif codice == 403:
            self.errore_ricevuto.emit("⚠ Accesso negato (403). Il modello potrebbe non essere accessibile.")
        elif codice == 429:
            self.errore_ricevuto.emit("⚠ Limite di richieste raggiunto (429). Riprova tra qualche secondo.")
        elif codice == 500:
            self.errore_ricevuto.emit("⚠ Errore interno del server (500). Riprova tra poco.")
        else:
            self.errore_ricevuto.emit(f"⚠ Errore HTTP {codice}: {dettaglio}")