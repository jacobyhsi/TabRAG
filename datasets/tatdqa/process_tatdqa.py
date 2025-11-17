import os
import shutil
import pymupdf  # PyMuPDF

# --------------------------------------------------------------
# PART 1 — PDF → per-page PNG conversion
# --------------------------------------------------------------

def process_pdfs_to_images(source_dir, dest_dir, dpi=300):
    os.makedirs(dest_dir, exist_ok=True)
    print(f"\nStarting PDF to image conversion from '{source_dir}'...")

    for filename in os.listdir(source_dir):
        if not filename.lower().endswith('.pdf'):
            continue

        source_path = os.path.join(source_dir, filename)
        base_filename = os.path.splitext(filename)[0]

        try:
            doc = pymupdf.open(source_path)
            print(f"  -> Processing '{filename}' ({len(doc)} pages)...")

            for i, page in enumerate(doc):
                page_num = i + 1
                page_base_name = f"{base_filename}_p{page_num}"

                # Per-page output folder
                page_folder = os.path.join(dest_dir, page_base_name)
                os.makedirs(page_folder, exist_ok=True)

                # Render page
                zoom = dpi / 72
                mat = pymupdf.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)

                # Save image
                output_filename = f"{page_base_name}.png"
                output_path = os.path.join(page_folder, output_filename)
                pix.save(output_path)

            doc.close()

        except Exception as e:
            print(f"  -> ⚠️ ERROR processing '{filename}': {e}")


# --------------------------------------------------------------
# PART 2 — Group generated images into batches of 25
# --------------------------------------------------------------

def group_images_for_retrieval(src_base, dst_base, group_size=25):
    os.makedirs(dst_base, exist_ok=True)

    all_images = []
    for root, _, files in os.walk(src_base):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                all_images.append(os.path.join(root, f))

    all_images.sort()

    for i in range(0, len(all_images), group_size):
        folder_idx = i // group_size
        folder_path = os.path.join(dst_base, f"folder{folder_idx}")
        os.makedirs(folder_path, exist_ok=True)

        batch = all_images[i:i+group_size]
        for img_path in batch:
            fname = os.path.basename(img_path)
            dst_path = os.path.join(folder_path, fname)
            shutil.copy(img_path, dst_path)

    print(f"Done. {len(all_images)} images distributed into "
          f"{(len(all_images) - 1) // group_size + 1} folders.")


# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------

if __name__ == "__main__":
    source_directory = 'datasets/tatdqa/test'
    generation_directory = 'datasets/tatdqa/generation'
    retrieval_directory = 'datasets/tatdqa/retrieval'

    # Step 1: Convert PDFs → images
    if not os.path.isdir(source_directory):
        print(f"Error: Source directory '{source_directory}' not found.")
    else:
        process_pdfs_to_images(source_directory, generation_directory, dpi=288)

        # Step 2: Collect & group images
        print("\nNow grouping images for retrieval...")
        group_images_for_retrieval(generation_directory, retrieval_directory, group_size=25)

        print("\nAll tasks completed.")
