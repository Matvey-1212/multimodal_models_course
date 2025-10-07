pip3 install wldhx.yadisk-direct
mkdir -p tags
mkdir -p indexes
curl -L $(yadisk-direct https://disk.360.yandex.ru/d/fEGBsQq_A45k4A) -o tags/openai_clip-vit-base-patch16__81444_tags_genre.npy
curl -L $(yadisk-direct https://disk.360.yandex.ru/d/h9eznn_HWDe4sw) -o tags/openai_clip-vit-base-patch16__81444_tags_style.npy

curl -L $(yadisk-direct https://disk.360.yandex.ru/d/9cIdqK3aglw6AA) -o indexes/Salesforce_blip-image-captioning-base_caption_81444.txt
curl -L $(yadisk-direct https://disk.360.yandex.ru/d/wdakb7Vscc1ZzQ) -o indexes/facebook_dinov2-base_81444.index
curl -L $(yadisk-direct https://disk.360.yandex.ru/d/dR37chysGri1mQ) -o indexes/google_siglip2-base-patch16-224_81444.index
curl -L $(yadisk-direct https://disk.360.yandex.ru/d/2Td_lSKIvffMww) -o indexes/openai_clip-vit-base-patch16_81444.index
curl -L $(yadisk-direct https://disk.360.yandex.ru/d/tj79OZOeJqOogw) -o indexes/openai_clip-vit-base-patch16_81444_text.index