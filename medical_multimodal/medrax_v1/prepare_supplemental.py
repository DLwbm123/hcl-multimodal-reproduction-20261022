"""Resolve the pinned CheXagent constructor's nested SigLIP dependency."""
import json
from prepare import ROOT, download, save


if __name__ == '__main__':
    plan = json.loads((ROOT / 'code/supplemental_assets.json').read_text())
    for item in plan['files']:
        download(item)
    save('SUPPLEMENTAL_PREPARED.json', {'download_bytes': plan['download_bytes'],
                                      'revision': plan['revision'], 'model_generations': 0})
