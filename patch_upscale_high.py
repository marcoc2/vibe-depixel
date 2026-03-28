#!/usr/bin/env python3
"""
patch_upscale_high.py — Patch ComfyUI metadata in upscale_fix HR images for upscale_high training.

Changes applied per PNG:
  1. Input node:
       - GIF case:    LoadGifFrames (329) → LoadImage loading the image itself
       - Sprite case: ImageGridSlicer (373) removed; upstream LoadImage (364) loads self
  2. Node 281 (ImageUpscaleWithModel): bypassed — i2i 1x, sem upscale
  3. Node 282 (ACN 1ª passada): strength=0.86, start_percent=0, end_percent=0.93
  4. Node 223 (ACN 2ª passada): strength=0.63, start_percent=0, end_percent=1.0
  5. Node 357 (StringConcatenate): "dataset/upscale_fix/" → "dataset/upscale_high/"

Atualiza tanto 'prompt' (grafo de execução) quanto 'workflow' (visualização).
"""
import json
import struct
import zlib
import copy
import time
import uuid
import urllib.request
import urllib.error
from pathlib import Path
import argparse


# ── Patch prompt (grafo de execução) ──────────────────────────────────────────

COMFY_OUTPUT = Path("F:/AppsCrucial/ComfyUI_phoenix3/ComfyUI/output")


def comfy_image_ref(image_path: Path) -> str:
    """Retorna 'subfolder/filename [output]' relativo à pasta output do ComfyUI."""
    try:
        rel = image_path.relative_to(COMFY_OUTPUT)
        return str(rel).replace("\\", "/") + " [output]"
    except ValueError:
        return image_path.name  # fallback: só o nome


def patch_prompt(prompt: dict, image_path: Path) -> dict:
    p = copy.deepcopy(prompt)
    filename = comfy_image_ref(image_path)

    # --- Caso 1: LoadGifFrames → LoadImage ---
    gif_nodes = [nid for nid, n in p.items()
                 if isinstance(n, dict) and n.get("class_type") == "LoadGifFrames"]
    for nid in gif_nodes:
        p[nid]["class_type"] = "LoadImage"
        p[nid]["inputs"] = {"image": filename, "upload": "image"}
        # Remover nós que referenciam slots 1, 2, 3 (frame_map, unique_count, total_frames)
        dead = [other for other, nd in p.items()
                if isinstance(nd, dict)
                and any(isinstance(v, list) and len(v) == 2
                        and v[0] == nid and v[1] in (1, 2, 3)
                        for v in nd.get("inputs", {}).values())]
        for d in dead:
            p.pop(d, None)

    # --- Caso 2: ImageGridSlicer — remover, redirecionar p/ LoadImage upstream ---
    slicer_nodes = [nid for nid, n in p.items()
                    if isinstance(n, dict) and n.get("class_type") == "ImageGridSlicer"]
    for nid in slicer_nodes:
        slicer_img_in = p[nid]["inputs"].get("image")  # ex: ["364", 0]
        if isinstance(slicer_img_in, list):
            src_nid = slicer_img_in[0]
            if src_nid in p:
                p[src_nid]["class_type"] = "LoadImage"
                p[src_nid]["inputs"] = {"image": filename, "upload": "image"}
            # Redirecionar consumidores do slicer para a fonte direta
            for other_nd in p.values():
                if not isinstance(other_nd, dict):
                    continue
                for k, v in other_nd.get("inputs", {}).items():
                    if isinstance(v, list) and len(v) == 2 and v[0] == nid and v[1] == 0:
                        other_nd["inputs"][k] = [src_nid, 0]
        p.pop(nid, None)

    # --- Re-patch: atualizar image ref em nós fonte já convertidos (idempotente) ---
    # Nós de entrada conhecidos: 329 (GIF) e 364 (sprite). Se já são LoadImage mas
    # com path errado (sem [output]), corrige.
    for src_nid in ("329", "364"):
        if src_nid in p and isinstance(p[src_nid], dict):
            nd = p[src_nid]
            if nd.get("class_type") == "LoadImage" and nd["inputs"].get("image") != filename:
                nd["inputs"]["image"] = filename
                nd["inputs"]["upload"] = "image"

    # --- Bypass nó 281 (ImageUpscaleWithModel) ---
    if "281" in p:
        upstream = p["281"]["inputs"].get("image")
        for other_nd in p.values():
            if not isinstance(other_nd, dict):
                continue
            for k, v in other_nd.get("inputs", {}).items():
                if isinstance(v, list) and len(v) == 2 and v[0] == "281" and v[1] == 0:
                    other_nd["inputs"][k] = upstream
        p.pop("281", None)

    # --- Nós 351/353 (MathExpression *4): mudar para *1 pois input já é HR ---
    for nid in ("351", "353"):
        if nid in p and isinstance(p[nid], dict):
            if p[nid].get("class_type") in ("MathExpression|pysssss", "MathExpression"):
                p[nid]["inputs"]["expression"] = "a * 1"

    # --- Bypass ImageListToBatch+ (acumulador de estado entre execuções) ---
    accum_nodes = [nid for nid, n in p.items()
                   if isinstance(n, dict) and "ImageListToBatch" in n.get("class_type", "")]
    for nid in accum_nodes:
        upstream = p[nid]["inputs"].get("image")
        for other_nd in p.values():
            if not isinstance(other_nd, dict):
                continue
            for k, v in other_nd.get("inputs", {}).items():
                if isinstance(v, list) and len(v) == 2 and v[0] == nid and v[1] == 0:
                    other_nd["inputs"][k] = upstream
        p.pop(nid, None)

    # --- Nó 282: strength=0.86, start_percent=0, end_percent=0.93 ---
    if "282" in p:
        p["282"]["inputs"]["strength"] = 0.86
        p["282"]["inputs"]["start_percent"] = 0.0
        p["282"]["inputs"]["end_percent"] = 0.93

    # --- Nó 223: strength=0.63, start_percent=0, end_percent=1.0 ---
    if "223" in p:
        p["223"]["inputs"]["strength"] = 0.63
        p["223"]["inputs"]["start_percent"] = 0.0
        p["223"]["inputs"]["end_percent"] = 1.0

    # --- Nó 357: output path (path dinâmico via StringConcatenate) ---
    if "357" in p:
        val = p["357"]["inputs"].get("string_a", "")
        if isinstance(val, str):
            p["357"]["inputs"]["string_a"] = (
                val.replace("dataset/upscale_fix/", "dataset/upscale_high/")
                   .replace("dataset/upscale/", "dataset/upscale_high/")
            )

    # --- Todos os SaveImage com filename_prefix hardcoded ---
    for nd in p.values():
        if not isinstance(nd, dict) or nd.get("class_type") != "SaveImage":
            continue
        prefix = nd["inputs"].get("filename_prefix", "")
        if isinstance(prefix, str):
            nd["inputs"]["filename_prefix"] = (
                prefix.replace("dataset/upscale_fix/", "dataset/upscale_high/")
                      .replace("dataset/upscale/", "dataset/upscale_high/")
            )

    return p


# ── Patch workflow (visualização no ComfyUI) ──────────────────────────────────

def patch_workflow(wf: dict, image_path: Path) -> dict:
    if not wf.get("nodes"):
        return wf  # sem dados de workflow para atualizar
    wf = copy.deepcopy(wf)
    filename = comfy_image_ref(image_path)

    nodes_by_id = {n["id"]: n for n in wf["nodes"]}
    # link format: [link_id, from_node_id, from_slot, to_node_id, to_slot, type]
    links_by_id = {l[0]: l for l in wf.get("links", [])}

    def remove_links(link_ids: set):
        """Remove links do array wf['links'] e desconecta nos receptores."""
        wf["links"] = [l for l in wf.get("links", []) if l[0] not in link_ids]
        for node in wf["nodes"]:
            for inp in node.get("inputs", []):
                if inp.get("link") in link_ids:
                    inp["link"] = None

    # --- Caso 1: LoadGifFrames → LoadImage ---
    for node in wf["nodes"]:
        if node.get("type") != "LoadGifFrames":
            continue
        # Links do slot 0 (IMAGE) ficam; slots 1-3 são removidos
        slot0_links = []
        dead_links = set()
        for out in node.get("outputs", []):
            idx = out.get("slot_index", node["outputs"].index(out))
            links = out.get("links") or []
            if idx == 0:
                slot0_links = links
            else:
                dead_links.update(links)
        remove_links(dead_links)

        node["type"] = "LoadImage"
        node["properties"] = {"cnr_id": "comfy-core", "ver": "0.16.4",
                               "Node name for S&R": "LoadImage"}
        node["widgets_values"] = [filename, "image"]
        node["inputs"] = []
        node["outputs"] = [
            {"name": "IMAGE", "type": "IMAGE", "links": slot0_links, "slot_index": 0},
            {"name": "MASK",  "type": "MASK",  "links": None},
        ]

    # --- Caso 2: ImageGridSlicer ---
    for node in list(wf["nodes"]):
        if node.get("type") != "ImageGridSlicer":
            continue

        # Link que conecta source → slicer (entrada "image" do slicer)
        slicer_in_link = None
        for inp in node.get("inputs", []):
            if inp["name"] == "image":
                slicer_in_link = inp.get("link")
                break

        src_id = None
        if slicer_in_link is not None:
            linfo = links_by_id.get(slicer_in_link)
            if linfo:
                src_id = linfo[1]

        # Links de saída do slicer (slot 0)
        slicer_out_links = []
        for out in node.get("outputs", []):
            slicer_out_links.extend(out.get("links") or [])

        # Redirecionar links de saída do slicer para partir do src_id
        if src_id is not None:
            for l in wf.get("links", []):
                if l[0] in slicer_out_links:
                    l[1] = src_id   # from_node agora é a fonte
                    l[2] = 0        # slot 0

            # Atualizar outputs da fonte
            src_node = nodes_by_id.get(src_id)
            if src_node:
                src_node["type"] = "LoadImage"
                src_node["widgets_values"] = [filename, "image"]
                src_node["properties"] = {"cnr_id": "comfy-core", "ver": "0.16.4",
                                          "Node name for S&R": "LoadImage"}
                src_node["inputs"] = []
                src_node["outputs"] = [
                    {"name": "IMAGE", "type": "IMAGE", "links": slicer_out_links, "slot_index": 0},
                    {"name": "MASK",  "type": "MASK",  "links": None},
                ]

        # Remover link slicer_in_link e o nó do slicer
        remove_links({slicer_in_link} if slicer_in_link else set())
        wf["nodes"] = [n for n in wf["nodes"] if n["id"] != node["id"]]
        wf.get("extra", {})  # no-op; slicer removido acima

    # Rebuild nodes_by_id após possíveis remoções
    nodes_by_id = {n["id"]: n for n in wf["nodes"]}

    # --- Nó 281 e ImageListToBatch+: bypass (mode=4) ---
    if 281 in nodes_by_id:
        nodes_by_id[281]["mode"] = 4
    for node in wf["nodes"]:
        if "ImageListToBatch" in node.get("type", ""):
            node["mode"] = 4

    # --- Nós 351/353: mudar expressão *4 → *1 ---
    for wid in (351, 353):
        if wid in nodes_by_id:
            wv = nodes_by_id[wid].get("widgets_values", [])
            if wv and isinstance(wv[0], str) and "*" in wv[0]:
                # widgets_values[0] é a expression
                nodes_by_id[wid]["widgets_values"][0] = "a * 1"

    # --- Nó 282: widgets_values [strength, start_percent, end_percent] ---
    if 282 in nodes_by_id:
        wv = nodes_by_id[282].get("widgets_values", [])
        if len(wv) >= 3:
            nodes_by_id[282]["widgets_values"] = [0.86, 0, 0.93] + list(wv[3:])

    # --- Nó 223 ---
    if 223 in nodes_by_id:
        wv = nodes_by_id[223].get("widgets_values", [])
        if len(wv) >= 3:
            nodes_by_id[223]["widgets_values"] = [0.63, 0, 1.0] + list(wv[3:])

    # --- Nó 357: output path ---
    if 357 in nodes_by_id:
        wv = nodes_by_id[357].get("widgets_values", [])
        if wv and isinstance(wv[0], str):
            wv[0] = (wv[0].replace("dataset/upscale_fix/", "dataset/upscale_high/")
                         .replace("dataset/upscale/", "dataset/upscale_high/"))

    return wf


# ── PNG I/O ───────────────────────────────────────────────────────────────────

def read_png_text_chunks(raw: bytes) -> dict:
    assert raw[:8] == b"\x89PNG\r\n\x1a\n", "Não é PNG"
    pos = 8
    chunks = {}
    while pos < len(raw):
        length = int.from_bytes(raw[pos:pos+4], "big")
        ctype = raw[pos+4:pos+8]
        cdata = raw[pos+8:pos+8+length]
        pos += 12 + length
        if ctype == b"tEXt":
            sep = cdata.index(b"\x00")
            key = cdata[:sep].decode("latin-1")
            chunks[key] = cdata[sep+1:].decode("latin-1")
    return chunks


def build_text_chunk(keyword: str, text: str) -> bytes:
    payload = keyword.encode("latin-1") + b"\x00" + text.encode("latin-1")
    ctype = b"tEXt"
    crc = zlib.crc32(ctype + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + ctype + payload + struct.pack(">I", crc)


def rewrite_png(raw: bytes, new_texts: dict) -> bytes:
    """Reescreve o PNG substituindo/adicionando os tEXt chunks especificados."""
    sig = b"\x89PNG\r\n\x1a\n"
    out = bytearray(sig)
    pos = 8
    written = set()
    while pos < len(raw):
        length = int.from_bytes(raw[pos:pos+4], "big")
        ctype = raw[pos+4:pos+8]
        cdata = raw[pos+8:pos+8+length]
        pos += 12 + length

        if ctype == b"tEXt":
            sep = cdata.index(b"\x00")
            key = cdata[:sep].decode("latin-1")
            if key in new_texts:
                if key not in written:
                    out += build_text_chunk(key, new_texts[key])
                    written.add(key)
                continue  # descarta chunk original

        # Chunk inalterado
        out += struct.pack(">I", length) + ctype + cdata
        out += struct.pack(">I", zlib.crc32(ctype + cdata) & 0xFFFFFFFF)

    # Escrever quaisquer chaves novas que não existiam antes
    for key, val in new_texts.items():
        if key not in written:
            out += build_text_chunk(key, val)

    return bytes(out)


def patch_png(path: Path, dry_run: bool = False) -> bool:
    raw = path.read_bytes()
    text_chunks = read_png_text_chunks(raw)

    if "prompt" not in text_chunks:
        return False

    prompt = json.loads(text_chunks["prompt"])
    new_prompt = patch_prompt(prompt, path)
    prompt_changed = new_prompt != prompt

    workflow = None
    new_workflow = None
    wf_changed = False
    if "workflow" in text_chunks:
        workflow = json.loads(text_chunks["workflow"])
        new_workflow = patch_workflow(workflow, path)
        wf_changed = new_workflow != workflow

    if not prompt_changed and not wf_changed:
        return False

    if dry_run:
        return True

    new_texts = {}
    if prompt_changed:
        new_texts["prompt"] = json.dumps(new_prompt, ensure_ascii=True, separators=(",", ":"))
    if wf_changed:
        new_texts["workflow"] = json.dumps(new_workflow, ensure_ascii=True, separators=(",", ":"))

    path.write_bytes(rewrite_png(raw, new_texts))
    return True


# ── Queue ─────────────────────────────────────────────────────────────────────

COMFY_URL = "http://127.0.0.1:8188"


def get_queue_size(url: str) -> int:
    try:
        with urllib.request.urlopen(f"{url}/queue") as r:
            d = json.loads(r.read())
            return len(d.get("queue_running", [])) + len(d.get("queue_pending", []))
    except Exception:
        return -1


def queue_prompt(prompt: dict, client_id: str, url: str) -> str | None:
    payload = json.dumps({"prompt": prompt, "client_id": client_id}).encode()
    req = urllib.request.Request(f"{url}/prompt", data=payload,
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read()).get("prompt_id")
    except urllib.error.HTTPError as e:
        print(f"    [HTTP {e.code}] {e.read().decode()[:200]}")
        return None


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Patch + queue metadata para upscale_high dataset")
    parser.add_argument(
        "folder",
        nargs="?",
        default="F:/AppsCrucial/ComfyUI_phoenix3/ComfyUI/output/dataset/upscale_fix_2nd",
        help="Pasta raiz (varre recursivamente apenas subpastas 'hr')",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Mostrar o que mudaria sem gravar nem enfileirar")
    parser.add_argument("--queue", action="store_true",
                        help="Após patchear, enfileirar todos no ComfyUI")
    parser.add_argument("--url", default=COMFY_URL, help="ComfyUI URL")
    parser.add_argument("--max-queue", type=int, default=4,
                        help="Pausar quando fila atingir esse tamanho (default: 4)")
    parser.add_argument("--delay", type=float, default=0.3,
                        help="Segundos entre envios (default: 0.3)")
    args = parser.parse_args()

    folder = Path(args.folder)
    pngs = sorted(p for p in folder.rglob("*.png") if p.parent.name == "hr")
    print(f"Scanning {len(pngs)} PNGs em {folder}")

    patched = skipped = errors = 0
    for p in pngs:
        try:
            changed = patch_png(p, dry_run=args.dry_run)
            if changed:
                patched += 1
                if args.dry_run:
                    print(f"  [would patch] {p.relative_to(folder)}")
            else:
                skipped += 1
        except Exception as e:
            errors += 1
            print(f"  [error] {p.name}: {e}")

    action = "Would patch" if args.dry_run else "Patched"
    print(f"\n{action}: {patched}  |  skipped: {skipped}  |  errors: {errors}")

    if not args.queue or args.dry_run:
        return

    # ── Enfileirar no ComfyUI ──
    print(f"\nEnfileirando {len(pngs)} jobs no ComfyUI ({args.url})...")
    client_id = str(uuid.uuid4())
    queued = q_errors = 0

    for i, p in enumerate(pngs, 1):
        try:
            from PIL import Image
            prompt = json.loads(Image.open(p).info["prompt"])
        except Exception as e:
            print(f"  [skip] {p.name}: {e}")
            continue

        while True:
            qs = get_queue_size(args.url)
            if qs == -1:
                print("  [erro] ComfyUI inacessível — está rodando?")
                return
            if qs < args.max_queue:
                break
            print(f"  [wait] fila={qs}, aguardando...", end="\r")
            time.sleep(5)

        pid = queue_prompt(prompt, client_id, args.url)
        if pid:
            queued += 1
            print(f"  [{i}/{len(pngs)}] {p.parent.parent.name}/{p.name}  id={pid[:8]}")
        else:
            q_errors += 1
            print(f"  [{i}/{len(pngs)}] FALHOU {p.name}")

        time.sleep(args.delay)

    print(f"\nQueued: {queued}  |  errors: {q_errors}")


if __name__ == "__main__":
    main()
