"""Tests of the interface languages (version 1.3): English, Russian, Chinese.

界面多语言测试：英语、俄语、中文。

The dictionaries of web/static/js/i18n.js are read as text: every key used by
index.html, app.js and charts.js must exist in all three languages, and every
algorithm parameter must have a Russian and a Chinese label and help text.
The last tests run i18n.js in Node.js (skipped when Node.js is not installed)
to check the placeholders and the translation of the server's phrases.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from maritime_route.routing.parameters import FIELDS

STATIC = Path(__file__).resolve().parents[1] / "src" / "maritime_route" / "web" / "static"
I18N_JS = STATIC / "js" / "i18n.js"
LANGS = ("en", "ru", "zh")


def _block(text: str, start: str, end: str) -> str:
    """Text between the first `start` and the next `end` after it."""
    i = text.index(start)
    return text[i:text.index(end, i + len(start))]


def _dictionaries() -> dict:
    """Keys of D.en, D.ru and D.zh."""
    text = I18N_JS.read_text(encoding="utf-8")
    d = _block(text, "const D = {", "const P = {")
    blocks = {"en": _block(d, "en: {", "\n    ru: {"),
              "ru": _block(d, "ru: {", "\n    zh: {"),
              "zh": d[d.index("zh: {"):]}
    return {lang: set(re.findall(r"'([\w.]+)':", body)) for lang, body in blocks.items()}


def _parameter_labels() -> dict:
    text = I18N_JS.read_text(encoding="utf-8")
    p = _block(text, "const P = {", "const UNITS = {")
    return {"ru": set(re.findall(r"^\s+(\w+): \[", _block(p, "ru: {", "zh: {"), re.M)),
            "zh": set(re.findall(r"^\s+(\w+): \[", p[p.index("zh: {"):], re.M))}


def _used_keys() -> set:
    """Keys used literally by the page and the scripts."""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    keys = set(re.findall(r'data-i18n(?:-html|-title)?="([\w.]+)"', html))
    for route in re.findall(r'data-route="([^"]+)"', html):
        keys.update("city." + c for c in route.split(","))
    for name in ("app.js", "charts.js"):
        js = (STATIC / "js" / name).read_text(encoding="utf-8")
        # Literal keys only: t('stage.' + name) builds the key at run time.
        keys.update(re.findall(r"\bt\('([\w.]*\w)'", js))
    return keys


def test_dictionaries_have_the_same_keys():
    d = _dictionaries()
    assert len(d["en"]) > 200
    for lang in ("ru", "zh"):
        assert d[lang] == d["en"], (lang, sorted(d["en"] - d[lang]), sorted(d[lang] - d["en"]))


def test_every_used_key_is_translated():
    d = _dictionaries()
    used = _used_keys()
    assert {"btn.plan", "btn.stop", "rf.copy", "city.Kiel"} <= used
    for lang in LANGS:
        assert not used - d[lang], (lang, sorted(used - d[lang]))


def test_page_loads_i18n_before_the_application():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert html.index("/static/js/i18n.js") < html.index("/static/js/api.js") < html.index(
        "/static/js/app.js")
    assert all(f'data-lang="{lang}"' in html for lang in LANGS)


@pytest.mark.parametrize("lang", ["ru", "zh"])
def test_every_parameter_has_a_label_and_help(lang):
    assert _parameter_labels()[lang] == set(FIELDS)


def test_outcome_and_stage_keys_exist_for_every_status():
    d = _dictionaries()
    for status in ("found", "no_path", "step_limit", "cancelled", "time_limit"):
        for lang in LANGS:
            assert f"status.{status}" in d[lang] and f"tag.{status}" in d[lang]


NODE = shutil.which("node")


def _run_node(script: str) -> dict:
    """Run i18n.js with a minimal browser stub and return the printed JSON."""
    stub = ("globalThis.localStorage={_s:{},getItem(k){return this._s[k]??null},"
            "setItem(k,v){this._s[k]=v}};globalThis.navigator={language:'en-US'};"
            "globalThis.document={querySelectorAll(){return []},documentElement:{},title:''};")
    code = stub + I18N_JS.read_text(encoding="utf-8") + "\n" + script
    out = subprocess.run([NODE, "-e", code], capture_output=True, text=True, timeout=60, check=True)
    return json.loads(out.stdout)


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_placeholders_and_language_switch():
    r = _run_node("""
      const a = t('params.changed', {n: 3});
      I18N.setLang('ru'); const b = t('params.changed', {n: 3});
      I18N.setLang('zh'); const c = t('params.changed', {n: 3});
      const missing = t('no.such.key');
      I18N.setLang('xx'); const still = I18N.lang;
      console.log(JSON.stringify({a, b, c, missing, still, zone: I18N.zone('OPEN_SEA')}));""")
    assert r == {"a": "3 changed", "b": "изменено: 3", "c": "已修改 3 项",
                 "missing": "no.such.key", "still": "zh", "zone": "开阔海域"}


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_server_phrases_are_translated():
    r = _run_node("""
      const msgs = [
        'Optimal path found; coastline check passed',
        'Step limit of 1,000 reached before the goal (search radius 397 weighted km) - increase the maximum number of steps',
        'Optimal path found; coastline check: the first route crossed land for 12 km and was corrected automatically (1 re-search with 4 cells blocked and a finer lattice (0.1°))',
        'Route through 1 intermediate point (2 legs): leg 1: Optimal path found; leg 2: Bellman equation solved in 40 sweeps',
        'something the client does not know'];
      const out = {};
      for (const lang of ['en', 'ru', 'zh']) { I18N.setLang(lang); out[lang] = msgs.map(I18N.server); }
      console.log(JSON.stringify(out));""")
    assert r["en"][0] == "Optimal path found; coastline check passed"
    assert r["ru"][0] == "Оптимальный путь найден; проверка по береговой линии пройдена"
    assert r["zh"][1] == "到达目标前已达到 1,000 步的上限（搜索半径 397 加权公里）——请增大最大步数"
    assert "повторных поисков: 1, заблокировано ячеек: 4 и более мелкая сетка (0.1°)" in r["ru"][2]
    assert r["zh"][3] == "经过 1 个途经点的航线（2 段）：第 1 段：已找到最优路径; 第 2 段：贝尔曼方程经 40 次迭代求解"
    assert r["ru"][4] == r["zh"][4] == "something the client does not know"
