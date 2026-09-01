import Markdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

interface MarkdownContentProps {
  readonly content: string;
}

const MARKDOWN_COMPONENTS: Components = {
  a({ node, ...properties }) {
    void node;
    return (
      <a
        {...properties}
        target="_blank"
        rel="noopener noreferrer"
      />
    );
  },
};

/** 安全渲染模型生成的 Markdown，不执行其中的原始 HTML。 */
export function MarkdownContent({ content }: MarkdownContentProps) {
  return (
    <div className="markdown-content">
      <Markdown
        remarkPlugins={[remarkGfm]}
        components={MARKDOWN_COMPONENTS}
        skipHtml
      >
        {content}
      </Markdown>
    </div>
  );
}
