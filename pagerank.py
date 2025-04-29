import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")

    corpus = crawl(sys.argv[1])

    # Sample PageRank
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print("PageRank Results from Sampling (n = 10000)")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")

    # Iterative PageRank
    ranks = iterate_pagerank(corpus, DAMPING)
    print("\nPageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a set of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r'<a\s+(?:[^>]*?)href="([^"]*)"', contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages

def transition_model(corpus, page, damping_factor):
    model = dict()
    total_pages = len(corpus)

    # Si la página tiene links salientes
    if corpus[page]:
        prob_link = damping_factor / len(corpus[page])
        prob_random = (1 - damping_factor) / total_pages
        for p in corpus:
            model[p] = prob_random
            if p in corpus[page]:
                model[p] += prob_link
    else:
        # Si la página no tiene links, se distribuye a todas las páginas igual
        for p in corpus:
            model[p] = 1 / total_pages

    return model

def sample_pagerank(corpus, damping_factor, n):
    import random

    page_rank = dict()
    for page in corpus:
        page_rank[page] = 0

    # Elegir una página inicial al azar
    page = random.choice(list(corpus.keys()))

    for _ in range(n):
        page_rank[page] += 1
        model = transition_model(corpus, page, damping_factor)
        pages = list(model.keys())
        probabilities = list(model.values())
        page = random.choices(pages, probabilities)[0]

    # Normalizar los valores para que sumen 1
    total = sum(page_rank.values())
    for page in page_rank:
        page_rank[page] /= total

    return page_rank


def iterate_pagerank(corpus, damping_factor):
    N = len(corpus)
    pagerank = dict()

    # Inicializar todos los PageRank en 1/N
    for page in corpus:
        pagerank[page] = 1 / N

    converged = False
    while not converged:
        new_pagerank = dict()
        converged = True

        for page in corpus:
            # Componente aleatorio
            new_rank = (1 - damping_factor) / N

            # Sumar contribuciones de cada página que enlaza a 'page'
            for possible_page in corpus:
                if page in corpus[possible_page] or not corpus[possible_page]:
                    num_links = len(corpus[possible_page]) if corpus[possible_page] else N
                    new_rank += damping_factor * (pagerank[possible_page] / num_links)

            new_pagerank[page] = new_rank

            # Verificar si el valor cambió más de 0.001
            if abs(new_pagerank[page] - pagerank[page]) > 0.001:
                converged = False

        pagerank = new_pagerank.copy()

    return pagerank

if __name__ == "__main__":
    main()
