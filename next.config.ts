import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  output: 'export', // 👈 add this line to enable static export
  // you can add other config options here
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
